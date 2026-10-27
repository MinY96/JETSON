2026-10-27 17:12:43,902 ERROR src.collectors.ftp.collector - [SMDI06] Recipe download failed: IL01_920562UM (1 consecutive failures)
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 420, in _scan_recipe
    safe_download_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 272, in safe_download_recipe
    ftp_download_tree(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 217, in ftp_download_tree
    ftp_download_tree(ftp, entry.name, target)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 217, in ftp_download_tree
    ftp_download_tree(ftp, entry.name, target)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 217, in ftp_download_tree
    ftp_download_tree(ftp, entry.name, target)
  [Previous line repeated 9 more times]
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 221, in ftp_download_tree
    with part.open("wb") as stream:
         ^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'D:\\DM_DATA_2\\SMDI06\\.IL01_920562UM.staging-63db2a695aa148c08d27efefa1edaf54\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\IL01_920562UM\\History.Log.part'
