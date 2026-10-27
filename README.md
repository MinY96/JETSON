2026-10-27 15:39:56,511 ERROR src.collectors.ftp.collector - [SMDI06] Recipe download failed: YV17_1565PAD3
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v2\src\collectors\ftp\collector.py", line 263, in _scan_recipe
    safe_download_recipe(ftp, entry.name, dest, required)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v2\src\collectors\ftp\client.py", line 88, in safe_download_recipe
    ftp_download_tree(ftp, recipe, stage)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v2\src\collectors\ftp\client.py", line 63, in ftp_download_tree
    entries = list_entries(ftp)
              ^^^^^^^^^^^^^^^^^
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v2\src\collectors\ftp\client.py", line 49, in list_entries
    ftp.retrlines("LIST", lines.append)
  File "C:\Users\10007166\AppData\Local\anaconda3\Lib\ftplib.py", line 461, in retrlines
    resp = self.sendcmd('TYPE A')
           ^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\anaconda3\Lib\ftplib.py", line 281, in sendcmd
    return self.getresp()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\anaconda3\Lib\ftplib.py", line 254, in getresp
    raise error_perm(resp)
ftplib.error_perm: 550 /d:/KT1025/Usr/Ins8/OD42_666562UM/YV17_1565PAD3: No such file or directory.
