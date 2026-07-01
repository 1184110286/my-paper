@echo off
call D:\anaconda3\Scripts\activate.bat mal
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
