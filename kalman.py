import pykalman
from collections import deque
from typing import Sequence, Tuple, Union 
import random

class OnlineUnivariateKalmanFilter:
  """
    An online kalman filter for a univariate time series. This periodically updates the
    internal state for every `update_interval` observations it's observed. It also
    maintains an internal window, which will compute mean/covariance updates using
    all of the observations in the window.

    This performs better than simply using the online method as described here:
    https://pykalman.github.io/kf_users_guide.html?#inferring-states

    Control the computational complexity by setting reasonable values for the
    internal window
  """

  def __init__(self, n_initialize: int = 25, update_interval: int = 500000, 
               internal_window: int=500, filter_type: str = 'normal',
               reparameterize: bool = True, **kwargs):
    
    _FILTER_TYPES = {
      'unscented': pykalman.UnscentedKalmanFilter,
      'unscented-additive': pykalman.AdditiveUnscentedKalmanFilter,
      'normal': pykalman.KalmanFilter
    }
    self.is_initialized = False
    self.n_initialize = n_initialize
    self.update_interval = update_interval
    self.internal_window = internal_window
    self.filter_type = filter_type
    self.reparameterize = reparameterize
    self.F = _FILTER_TYPES[filter_type](**kwargs)
    self.observations = deque(maxlen=internal_window)
    self.observed = 0


  def initialize(self, X: Sequence, n_iter: int = 10) -> None:
    """
    Initializes the kalman filter, setting the parameters 
    using expectation-maximization as defined in
    https://pykalman.github.io/#pykalman.KalmanFilter.em

    Parameters
    ----------
    X : Sequence
        a sequence of observations, must be at least n_initialize in length
    n_iter : int
        number of iterations to run the em algorithm, default = 5
    """
    if len(X) < self.n_initialize:
      raise Exception(f"need at least {self.n_initialize} observations to initialize the filter")
    
    if self.filter_type == "normal":
      self.F.em(X, n_iter)
    
    for val in X:
      self.observations.append(val)
    self.is_initialized = True

  def _em(self):
    """
    runs expectation-maximization on the internal window
    if the kalman filter type is normal
    """
    if self.filter_type == "normal":
      self.F.em(self.observations)

  def add_observation(self, X: Union[float, Sequence]) -> None:
    """
    adds an observation to this filter's internal window but
    does not produce an update. Use this to "warm up" the
    filter in an online setting
    
    Parameters
    ----------
    X : Union[float, Sequence]
        a single observation
    
    """
    self.observations.append(X)

  def observe(self, X: Union[float, Sequence], is_window: bool = False) -> Tuple[float, float]:
    """
    'observes' a time window of length n to produce the filtered result
    """
    means, covs = None, None
    if not self.is_initialized:
      raise Exception("Initialize the filter by running `initialize()`")
    else:
      if is_window:
        self.observations.append(X[-1])
        means, covs = self.F.filter(X)
      else:
        self.observations.append(X)
        means, covs = self.F.filter(self.observations)
    
    self.observed += 1

    if self.observed % self.update_interval == 0:
      if self.reparameterize:
        print("reiniting")
        self._em()

    next_mean, next_cov = self.F.filter_update(means[-1], covs[-1])

    return next_mean, next_cov

if __name__ == "__main__":
  data = [20,21,23,19,18,15,25,29,20,21,23,19,18,15,25,29,20,21,23,19,18,15,25,29,20,21,23,19,18,15,25,29,20,21,23,19,18,15,25,29,20,21,23,19,18,15,25,29]
  kal = OnlineUnivariateKalmanFilter()
  kal.initialize(data)
  kal._em()
  for i in range(200):
    if(i<25):
      mean,cov = kal.observe(20)
      print(mean)
    else:
      mean,cov = kal.observe(random.randint(20, 100))
      print(mean)  

