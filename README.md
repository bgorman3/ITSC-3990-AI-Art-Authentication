# ITSC-3990-AI-Art-Authentication
New Research Project for ITSC 3990 Using AI to Authenticate Art


Notes Lower Epochs 

PS C:\Users\getin\Documents\networking\ITSC-3990-AI-Art-Authentication> python train.py
>>
Training Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 317/317 [26:59<00:00,  5.11s/batch] 
Epoch 1/10, Loss: 0.0110
Validation Epoch 1/10: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 80/80 [02:27<00:00,  1.84s/batch]
Validation Accuracy: 1.0000
Training Epoch 2/10:  22%|████████████████████████████████████████████████▎                                                                                                                                                                          | 70/317 [05:53<20:47,  5.05s/batch]
Traceback (most recent call last):
  File "C:\Users\getin\Documents\networking\ITSC-3990-AI-Art-Authentication\train.py", line 51, in <module>
    train_model(model, train_loader, val_loader)
  File "C:\Users\getin\Documents\networking\ITSC-3990-AI-Art-Authentication\train.py", line 22, in train_model
    outputs = model(images)
              ^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\getin\AppData\Roaming\Python\Python312\site-packages\torchvision\models\resnet.py", line 285, in forward
    return self._forward_impl(x)
           ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\getin\AppData\Roaming\Python\Python312\site-packages\torchvision\models\resnet.py", line 275, in _forward_impl
    x = self.layer3(x)
        ^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\container.py", line 219, in forward
    input = module(input)
            ^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\getin\AppData\Roaming\Python\Python312\site-packages\torchvision\models\resnet.py", line 92, in forward
    out = self.conv1(x)
          ^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\conv.py", line 458, in forward
    return self._conv_forward(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Python312\Lib\site-packages\torch\nn\modules\conv.py", line 454, in _conv_forward
    return F.conv2d(input, weight, bias, self.stride,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
PS C:\Users\getin\Documents\networking\ITSC-3990-AI-Art-Authentication> 

Last Run 

Notes Save data to csv to upload to matlab to get the visualization of the data

Include saving of the data so if It gets interrupted again I won't lose all the progress. 