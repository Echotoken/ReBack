# ReBack
The official implementation of the paper "Need for Speed: Taming Backdoor Attacks with Speed and Precision" accepted by 45-th IEEE Symposium on Security and Privacy.

We provide ReBack demos based on [repository Fight-Poison-With-Poison](https://github.com/Unispac/Fight-Poison-With-Poison) from @Unispac.
First, put "reback" file in folder "other_cleansers" and then, call it in file "other_cleanser.py" by:
```
from other_cleansers import reback
reback.main(args, model, poisoned_set, num_classes, K=100)
```
