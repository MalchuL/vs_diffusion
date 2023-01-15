from torchmetrics import MetricCollection, MeanAbsoluteError, MeanSquaredLogError, PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure


def get_alignment_metrics(prefix='val/'):
    collection = MetricCollection(prefix=prefix,
                                  metrics={'L1': MeanAbsoluteError(),
                                           'L2': MeanSquaredLogError(),
                                           'PSNR': PeakSignalNoiseRatio(),
                                           'MSSIM': MultiScaleStructuralSimilarityIndexMeasure()})
    return collection
