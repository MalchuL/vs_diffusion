from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from git import Repo
import os
import shutil
from src import utils
from natsort import natsorted

log = utils.get_pylogger(__name__)


class GitDiffSaver(Callback):
    OUTPUT_FOLDER = "git_info"

    def __init__(self, repo_dir, tracking_files, save_untracked=False):
        self.repo_dir = repo_dir
        self.repo = Repo(self.repo_dir)
        self.tracking_files = tracking_files
        self.save_untracked = save_untracked

    @rank_zero_only
    def dump_git_data(self):
        hcommit = self.repo.head.commit
        git_diff = hcommit.diff(None)
        git_status_data = []

        modifiers = {'M': 'Modified', 'A': 'Added', 'R': 'Renamed'}
        if self.save_untracked:
            modifiers['U'] = 'Untracked'
        log.info(f'Following directories annd files are tracker {self.tracking_files}')
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)
        for mod, mod_name in modifiers.items():
            for d in git_diff.iter_change_type(mod):
                skip = True
                for rq in self.tracking_files:
                    if d.a_path is not None and d.a_path.startswith(rq):
                        log.debug(f'Matching: {rq}: {d.a_path}')
                        skip = False
                    elif d.b_path is not None and d.b_path.startswith(rq):
                        log.debug(f'Matching: {rq}: {d.b_path}')
                        skip = False
                if skip:
                    continue

                if mod == 'R':
                    path = d.b_path
                    git_status_data.append(f'{mod_name:9}:  {d.a_path} -> {d.b_path}')
                else:
                    path = d.a_path
                    git_status_data.append(f'{mod_name:9}:  {d.a_path}')

                os.makedirs(os.path.join(self.OUTPUT_FOLDER, 'modified_files', os.path.dirname(path)), exist_ok=True)
                if os.path.exists(os.path.join(self.repo_dir, path)):
                    shutil.copy(os.path.join(self.repo_dir, path), os.path.join(self.OUTPUT_FOLDER, 'modified_files', path))
                log.info(f'{path} saved into {self.OUTPUT_FOLDER} folder')

        with open(os.path.join(self.OUTPUT_FOLDER, 'git_status.txt'), 'w') as f:
            f.write('\n'.join(git_status_data))
        with open(os.path.join(self.OUTPUT_FOLDER, 'git_diff.txt'), 'w') as f:
            f.write(str(self.repo.git.diff(hcommit)))
        with open(os.path.join(self.OUTPUT_FOLDER, 'git_revision.txt'), 'w') as f:
            f.write(str(hcommit) + '\n')
            f.write(str(self.repo.git.status(hcommit)).split('\n')[0])

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.dump_git_data()
