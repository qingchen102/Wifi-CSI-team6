import time, random, shutil, os
from libsvm.svmutil import svm_load_model, svm_predict
src = r"D:\Wifi CSI\实验工作区\data\model\CSI_presence_final.model"
dst = r"D:\wifi_csi_work\model\CSI_presence_final.model"
os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(src, dst)
t0=time.time(); m=svm_load_model(dst); tload=time.time()-t0
x={i:random.random() for i in range(1,60)}
svm_predict([1],[x],m,'-q')
t0=time.time(); N=2000
for _ in range(N): svm_predict([1],[x],m,'-q')
dt=(time.time()-t0)/N
print("load %.2fs  predict %.3f ms/window  -> %.0f windows/s" % (tload, dt*1000, 1/dt))
print("model size %.2f MB" % (os.path.getsize(dst)/1048576))
