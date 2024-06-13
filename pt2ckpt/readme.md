* 本文件夹是执行权重转换，以及运行pt2ckpt.py和start_conversion.sh所必须的相关文件和指南。

### 1. 首先要实行权重转换必须要有：
* 原始的网络nets、问题problems、工具包utils文件夹用于torch.load。
* torch的相关环境，一般只需要安装torch（2.2）即可。

### 2. 准备好运行环境后，将本文件夹下`nets`、`problems`、`utils`三个文件夹和`pt2ckpt.py`、`start_conversion.sh`两个文件放入项目根目录，即`APSS/`下面

### 3. 然后执行`start_conversion.sh`即可，实际执行时，需要修改具体转换的pt文件名等参数，转换后的ckpt会保存再根目录的`checkpoint`文件夹下。



--------------------------
## 特别的，在云集服务器248及249上执行上述步骤，直接使用249上的pt2ckpt docker环境即可
* 可能需要的挂载：
    * 在248服务器上要挂载欧博原来在249上的torch代码：udo mount -t nfs 211.83.111.249:/data01/oujie/crlsf-pp /home/oj/distributed_floder/128_share_data01_oj/crlsf-pp
    * 在249服务器上挂载我们在248服务上的APSS代码：sudo mount -t nfs 211.83.111.248:/data01/cym/MindSpore /home/upa1/cym/MindSpore
