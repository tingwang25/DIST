class StandardScaler():
    r"""
    Description:
    -----------
    Standard the input.

    Args:
    -----------
    mean: float
        Mean of data.
    std: float
        Standard of data.

    Attributes:
    -----------
    Same as Args.
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean