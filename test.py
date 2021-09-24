# import tensorflow as tf
import jax
from jax.lib import xla_bridge as xb
from jaxlib import xla_client as xc
# from jaxlib.xla_client import _ipu as ipu
import jax.numpy as np

# from tensorflow.python import ipu
from jaxlib.xla_client import _ipu as ipu
# from tensorflow.python.ipu.config import IPUConfig
# from tensorflow.python.ipu.scopes import ipu_scope

# temp = 24
# devices = xb.local_devices(0, "ipu")
# device = devices[temp]

cfg = ipu.config.IPUConfig()
# cfg.auto_select_ipus = 1
cfg.select_ipus = 32
cfg.configure_ipu_system()
#ipu.utils.configure_ipu_system(cfg)

from jax import device_put
# a = np.array([[1.0, 0.0], [0.0, 1.0]])
# b = np.array([[2.0, 3.0], [1.0, 5.0]])
# # a = device_put(a,device)
# # b = device_put(b,device)
# a = device_put(a)
# b = device_put(b)
# c = np.dot(a,b)
# print(c)


strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[2.0, 3.0], [1.0, 5.0]])
    # a = device_put(a,device)
    # b = device_put(b,device)
    a = device_put(a)
    b = device_put(b)
    c = np.dot(a,b)
    print(c)
    # host_id = xb.host_id()
    # xc._ipu_backend_factory()
    # a = np.array([[1.0, 0.0], [0.0, 1.0]])
    # b = np.array([[2.0, 3.0], [1.0, 5.0]])
    # a = device_put(a,device)
    # b = device_put(b,device)
    # c = np.dot(a,b)
    # a = 1.0
    # b = 2.0
    # c = a+b
    # print(c)

# host_id = xb.host_id()
# xc._ipu_backend_factory()
# devices = xb.local_devices(0, "ipu")
# device = devices[0]

# a = device_put(a,device)
# b = device_put(b,device)

# c = np.dot(a,b)

# print(c)


















# from tensorflow.python import ipu
# from tensorflow.python.ipu.config import IPUConfig
# from tensorflow.python.ipu.scopes import ipu_scope

# # Configure argument for targeting the IPU
# cfg = IPUConfig()
# cfg.auto_select_ipus = 1
# cfg.configure_ipu_system()

# ipu.utils.configure_ipu_system(cfg)

# print(xb.get_backend("ipu"))

# xb.get_device_backend()
# print(jax.host_id())
# print(jax.host_ids())


# def dot(a,b):  # Define a function
#   return np.dot(a, b)

# Create an IPU distribution strategy.
# strategy = ipu.ipu_strategy.IPUStrategyV1()

# with strategy.scope():
#     print(tf.config.list_physical_devices("IPU"))
#     x = numpy.random.normal(size=(size, size)).astype(np.float32)
#     x = device_put(x)
    # print(dot(np.array([[1.0, 0.0], [0.0, 1.0]]),
    #       np.array([[4.0, 1.0], [2.0, 2.0]])))
    
# Create an IPU distribution strategy.
# temp = 32
# strategy = ipu.ipu_strategy.IPUStrategyV1()
# with strategy.scope():
#     host_id = xb.host_id()
#     xc._ipu_backend_factory()
#     devices = xb.local_devices(0, "ipu")
#     device = devices[temp]
#     a = device_put(a,device)
#     b = device_put(b,device)
#     c = np.dot(a,b)
#     print(c)

