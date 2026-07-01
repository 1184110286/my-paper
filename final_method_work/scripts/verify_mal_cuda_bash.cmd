@echo off
call D:\anaconda3\Scripts\activate.bat mal
"C:\Program Files\Git\bin\bash.exe" -lc "export PATH=/d/anaconda3/envs/mal:/d/anaconda3/envs/mal/Scripts:$PATH && which python && python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())'"
