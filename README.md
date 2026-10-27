(.venv) D:\개발\5_DM_RMS\3_dm_backup\fab1_v3>python run.py
2026-10-27 17:45:15,793 INFO src.collectors.ftp.collector - [SMDI01] FTP collection started
2026-10-27 17:45:34,585 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 1건째: A003_145062UM - 필수 Recipe 파일 없음: A003_145062UM/ins.dat
2026-10-27 17:45:43,657 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 2건째: A102_056562UM - 필수 Recipe 파일 없음: A102_056562UM/ins.dat
2026-10-27 17:45:44,345 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 3건째: A102_1439TRET - 필수 Recipe 파일 없음: A102_1439TRET/ins.dat
2026-10-27 17:45:48,199 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 4건째: A102_358062UM - 필수 Recipe 파일 없음: A102_358062UM/ins.dat
2026-10-27 17:47:35,422 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 5건째: AKLA_MDI03ORG - 필수 Recipe 파일 없음: AKLA_MDI03ORG/ins.dat
2026-10-27 17:48:07,844 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 6건째: AN32_694562UM - 필수 Recipe 파일 없음: AN32_694562UM/ins.dat
2026-10-27 17:48:19,813 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 7건째: AN34_4580SWET - 필수 Recipe 파일 없음: AN34_4580SWET/ins.dat
2026-10-27 17:51:28,782 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 8건째: BL08_147062UM - 필수 Recipe 파일 없음: BL08_147062UM/ins.dat
2026-10-27 17:55:49,729 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 9건째: CQ16_168462UM - 필수 Recipe 파일 없음: CQ16_168462UM/ins.dat
2026-10-27 17:56:13,024 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=BI19 (2 values)
2026-10-27 17:56:13,151 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 17:59:26,027 WARNING src.collectors.ftp.collector - [SMDI01] Recipe 제외(필수 파일 없음) 10건째: DJ01_366462UM - 필수 Recipe 파일 없음: DJ01_366462UM/ins.dat
2026-10-27 18:06:55,718 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 18:07:25,883 ERROR src.collectors.ftp.collector - [SMDI01] Recipe download failed: DO79_9200POPT (1 consecutive failures)
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 420, in _scan_recipe
    safe_download_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 272, in safe_download_recipe
    ftp_download_tree(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 222, in ftp_download_tree
    ftp.retrbinary(f"RETR {entry.name}", stream.write)
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 436, in retrbinary
    with self.transfercmd(cmd, rest) as conn:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 393, in transfercmd
    return self.ntransfercmd(cmd, rest)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 359, in ntransfercmd
    resp = self.sendcmd(cmd)
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 281, in sendcmd
    return self.getresp()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 254, in getresp
    raise error_perm(resp)
ftplib.error_perm: 550 /d:/KT0931/Usr/ins8/DO79_9200POPT/ins.dat: No such file or directory.
2026-10-27 18:18:07,635 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=DPE1 (2 values)
2026-10-27 18:18:07,754 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 18:29:09,577 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=E3NI (2 values)
2026-10-27 18:29:09,577 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=E3SB (2 values)
2026-10-27 18:29:09,689 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 18:39:34,965 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 18:49:54,386 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:00:27,500 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:10:23,132 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:20:49,243 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=M7L0 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=M7L1 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD03 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD08 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD09 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD10 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD11 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD13 (2 values)
2026-10-27 19:31:47,696 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD14 (2 values)
2026-10-27 19:31:47,807 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:41:58,235 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 19:52:36,089 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:02:39,318 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:12:40,214 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:22:40,164 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:32:49,731 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:42:38,322 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=SW7R (2 values)
2026-10-27 20:42:38,322 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=SW7X (2 values)
2026-10-27 20:42:38,449 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 20:53:23,142 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 21:03:59,863 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 21:13:47,175 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 21:24:06,589 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=WWA8 (2 values)
2026-10-27 21:24:06,725 INFO src.collectors.ftp.collector - [SMDI01] flushed 999 rows
2026-10-27 21:32:40,731 INFO src.collectors.ftp.collector - [SMDI01] flushed 827 rows
2026-10-27 21:32:40,731 INFO src.collectors.ftp.collector - [SMDI01] 수집 제외: 필수 파일 없음 86건, 경로 없음 0건
2026-10-27 21:32:40,731 ERROR src.collectors.ftp.collector - [SMDI01] FTP collection failed; other equipment continues
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 831, in run
    collect_equip_data(equip, equip_dir)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 786, in collect_equip_data
    raise RuntimeError(
RuntimeError: [SMDI01] 1 downloads failed; delete/update-history skipped; pending parses remain available
2026-10-27 21:32:40,748 INFO src.collectors.ftp.collector - [SMDI02] FTP collection started
2026-10-27 21:33:16,780 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 1건째: A004_7655M4ET - 필수 Recipe 파일 없음: A004_7655M4ET/ins.dat
2026-10-27 21:43:28,781 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=BI19 (2 values)
2026-10-27 21:43:28,909 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 21:54:24,046 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:05:28,454 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=DPE1 (2 values)
2026-10-27 22:05:28,581 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:16:29,879 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=E3NI (2 values)
2026-10-27 22:16:29,879 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=E3SB (2 values)
2026-10-27 22:16:30,006 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:27:08,133 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:37:28,565 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:47:18,894 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 2건째: JN05_1565PAD3 - 필수 Recipe 파일 없음: JN05_1565PAD3/ins.dat
2026-10-27 22:48:11,454 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 22:54:11,402 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 3건째: KT03_7073125U - 필수 Recipe 파일 없음: KT03_7073125U/ins.dat
2026-10-27 22:55:20,699 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 4건째: KT42_4580SWET - 필수 Recipe 파일 없음: KT42_4580SWET/ins.dat
2026-10-27 22:58:16,800 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 5건째: KV03_3365GATE - 필수 Recipe 파일 없음: KV03_3365GATE/ins.dat
2026-10-27 22:58:49,920 INFO src.collectors.ftp.collector - [SMDI02] flushed 999 rows
2026-10-27 23:02:24,452 ERROR src.collectors.ftp.collector - [SMDI02] Recipe download failed: LA09_5800CTPT (1 consecutive failures)
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 420, in _scan_recipe
    safe_download_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 272, in safe_download_recipe
    ftp_download_tree(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 222, in ftp_download_tree
    ftp.retrbinary(f"RETR {entry.name}", stream.write)
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 436, in retrbinary
    with self.transfercmd(cmd, rest) as conn:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 393, in transfercmd
    return self.ntransfercmd(cmd, rest)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 359, in ntransfercmd
    resp = self.sendcmd(cmd)
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 281, in sendcmd
    return self.getresp()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 254, in getresp
    raise error_perm(resp)
ftplib.error_perm: 550 /d:/Kt0933/Usr/Ins8/LA09_5800CTPT/lowds.img: No such file or directory.
2026-10-27 23:02:38,941 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 6건째: LC14_516062UM - 필수 Recipe 파일 없음: LC14_516062UM/ins.dat
2026-10-27 23:02:41,565 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 7건째: LC14_537562UM - 필수 Recipe 파일 없음: LC14_537562UM/ins.dat
2026-10-27 23:02:44,189 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 8건째: LC14_6400V1PT - 필수 Recipe 파일 없음: LC14_6400V1PT/ins.dat
2026-10-27 23:02:49,173 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 9건째: LC14_6600M2PT - 필수 Recipe 파일 없음: LC14_6600M2PT/ins.dat
2026-10-27 23:02:51,798 WARNING src.collectors.ftp.collector - [SMDI02] Recipe 제외(필수 파일 없음) 10건째: LC18_020162UM - 필수 Recipe 파일 없음: LC18_020162UM/ins.dat
2026-10-27 23:07:20,181 WARNING src.collectors.ftp.collector - [SMDI02] Recipe FTP 경로 이상, 새 연결로 1회 재시도: LC47_5085COSI - FTP 부모 폴더 복귀 실패: remote='LC47_5085COSI', parent='/d:/Kt0933/Usr/Ins8'
2026-10-27 23:07:41,214 ERROR src.collectors.ftp.collector - [SMDI02] FTP collection failed; other equipment continues
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 193, in ftp_download_tree
    names = ftp.nlst()
            ^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 547, in nlst
    self.retrlines(cmd, files.append)
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 459, in retrlines
    with self.transfercmd(cmd) as conn, \
         ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 393, in transfercmd
    return self.ntransfercmd(cmd, rest)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 359, in ntransfercmd
    resp = self.sendcmd(cmd)
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 281, in sendcmd
    return self.getresp()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 244, in getresp
    resp = self.getmultiline()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 230, in getmultiline
    line = self.getline()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 212, in getline
    line = self.file.readline(self.maxline + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\socket.py", line 720, in readinto
    return self._sock.recv_into(b)
           ^^^^^^^^^^^^^^^^^^^^^^^
TimeoutError: timed out

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 228, in ftp_download_tree
    ftp.cwd("..")
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 612, in cwd
    return self.voidcmd('CDUP')
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 286, in voidcmd
    return self.voidresp()
           ^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 259, in voidresp
    resp = self.getresp()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 244, in getresp
    resp = self.getmultiline()
           ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 230, in getmultiline
    line = self.getline()
           ^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 212, in getline
    line = self.file.readline(self.maxline + 1)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\socket.py", line 717, in readinto
    raise OSError("cannot read from timed out object")
OSError: cannot read from timed out object

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 596, in collect_kla2139
    _scan_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 420, in _scan_recipe
    safe_download_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 272, in safe_download_recipe
    ftp_download_tree(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 231, in ftp_download_tree
    raise FTPDirectoryError(
src.collectors.ftp.client.FTPDirectoryError: FTP 부모 폴더 복귀 실패: remote='LC47_5085COSI', parent='/d:/Kt0933/Usr/Ins8'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 831, in run
    collect_equip_data(equip, equip_dir)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 759, in collect_equip_data
    collect_kla2139(equip, equip_dir, lookup, state)
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 620, in collect_kla2139
    ftp = _connect_kla2139(equip)
          ^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 562, in _connect_kla2139
    ftp.connect(equip.ftp_info.host, equip.ftp_info.port)
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\ftplib.py", line 158, in connect
    self.sock = socket.create_connection((self.host, self.port), self.timeout,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\socket.py", line 865, in create_connection
    raise exceptions[0]
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\socket.py", line 850, in create_connection
    sock.connect(sa)
TimeoutError: [WinError 10060] 연결된 구성원으로부터 응답이 없어 연결하지 못했거나, 호스트로부터 응답이 없어 연결이 끊어졌습니다
2026-10-27 23:07:41,448 INFO src.collectors.ftp.collector - [SMDI03] FTP collection started
2026-10-27 23:18:23,498 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-27 23:29:24,088 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-27 23:40:52,008 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-27 23:51:52,066 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=DPE1 (2 values)
2026-10-27 23:51:52,066 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=E3NI (2 values)
2026-10-27 23:51:52,193 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:02:05,907 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:12:28,770 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:23:09,763 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:31:06,596 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 1건째: KT03_7073125U - 필수 Recipe 파일 없음: KT03_7073125U/ins.dat
2026-10-28 00:32:20,044 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 2건째: KT42_4580SWET - 필수 Recipe 파일 없음: KT42_4580SWET/ins.dat
2026-10-28 00:33:51,647 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:44:34,749 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=M7L0 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=M7TH (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD03 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD08 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD09 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD10 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD11 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD13 (2 values)
2026-10-28 00:55:25,650 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD14 (2 values)
2026-10-28 00:55:25,777 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:05:55,349 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:14:03,430 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 3건째: NS46_5085COSI - 필수 Recipe 파일 없음: NS46_5085COSI/ins.dat
2026-10-28 01:16:45,686 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=NS60 (2 values)
2026-10-28 01:16:45,813 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:27:19,982 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:37:50,396 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:48:36,103 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 01:59:20,699 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=SW7R (2 values)
2026-10-28 01:59:20,699 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=SW7X (2 values)
2026-10-28 01:59:20,824 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:10:28,409 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:18:47,445 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 4건째: UH01_147062UM - 필수 Recipe 파일 없음: UH01_147062UM/ins.dat
2026-10-28 02:21:09,424 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:22:16,310 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 5건째: UY04_169662UM - 필수 Recipe 파일 없음: UY04_169662UM/ins.dat
2026-10-28 02:31:04,315 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:41:27,692 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=WWA8 (2 values)
2026-10-28 02:41:27,819 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:45:05,633 WARNING src.collectors.ftp.collector - [SMDI03] Recipe 제외(필수 파일 없음) 6건째: YI01_5905125U - 필수 Recipe 파일 없음: YI01_5905125U/ins.dat
2026-10-28 02:51:54,462 INFO src.collectors.ftp.collector - [SMDI03] flushed 999 rows
2026-10-28 02:51:55,037 INFO src.collectors.ftp.collector - [SMDI03] flushed 1 rows
2026-10-28 02:51:55,037 INFO src.collectors.ftp.collector - [SMDI03] 수집 제외: 필수 파일 없음 6건, 경로 없음 0건
2026-10-28 02:51:55,123 INFO src.collectors.ftp.collector - [SMDI03] new=20980 update=0 removed=0 unchanged=0
2026-10-28 02:51:55,123 INFO src.collectors.ftp.collector - [SMDI08] FTP collection started
2026-10-28 02:56:48,506 WARNING src.collectors.ftp.collector - [SMDI08] Recipe 제외(필수 파일 없음) 1건째: BARE_SISI25UM - 필수 Recipe 파일 없음: BARE_SISI25UM/Sensitivity1.xml
2026-10-28 03:14:54,504 INFO src.collectors.ftp.collector - [SMDI08] flushed 999 rows
2026-10-28 03:17:45,565 WARNING src.collectors.ftp.collector - [SMDI08] Recipe 제외(필수 파일 없음) 2건째: EKAC_145062UM - 필수 Recipe 파일 없음: EKAC_145062UM/Sensitivity1.xml
2026-10-28 03:39:24,148 INFO src.collectors.ftp.collector - [SMDI08] flushed 999 rows
2026-10-28 03:44:45,668 WARNING src.collectors.ftp.collector - [SMDI08] Recipe 제외(필수 파일 없음) 3건째: MF01_652520UM - 필수 Recipe 파일 없음: MF01_652520UM/Sensitivity1.xml
2026-10-28 04:03:40,590 INFO src.collectors.ftp.collector - [SMDI08] flushed 999 rows
2026-10-28 04:10:50,766 WARNING src.collectors.ftp.collector - [SMDI08] Recipe 제외(필수 파일 없음) 4건째: NS82_487062UM - 필수 Recipe 파일 없음: NS82_487062UM/Sensitivity1.xml
2026-10-28 04:27:50,930 INFO src.collectors.ftp.collector - [SMDI08] flushed 999 rows
2026-10-28 04:51:26,192 INFO src.collectors.ftp.collector - [SMDI08] flushed 999 rows
2026-10-28 04:53:01,944 INFO src.collectors.ftp.collector - [SMDI08] flushed 63 rows
2026-10-28 04:53:01,944 INFO src.collectors.ftp.collector - [SMDI08] 수집 제외: 필수 파일 없음 4건, 경로 없음 0건
2026-10-28 04:53:01,960 INFO src.collectors.ftp.collector - [SMDI08] new=5058 update=0 removed=0 unchanged=0
2026-10-28 04:53:01,960 INFO src.collectors.ftp.collector - [SMPW06] FTP collection started
2026-10-28 05:12:48,583 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 05:33:20,861 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 05:54:58,152 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 06:01:28,785 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 1건째: JJ02_1565SL10 - 필수 Recipe 파일 없음: JJ02_1565SL10/Sensitivity1.xml
2026-10-28 06:01:31,245 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 2건째: JJ02_2000SL10 - 필수 Recipe 파일 없음: JJ02_2000SL10/Sensitivity1.xml
2026-10-28 06:01:31,404 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 3건째: JJ02_2240SL10 - 필수 Recipe 파일 없음: JJ02_2240SL10/Sensitivity1.xml
2026-10-28 06:01:46,996 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 4건째: JJ02_6065SL10 - 필수 Recipe 파일 없음: JJ02_6065SL10/Sensitivity1.xml
2026-10-28 06:01:47,156 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 5건째: JJ02_6066SL10 - 필수 Recipe 파일 없음: JJ02_6066SL10/Sensitivity1.xml
2026-10-28 06:01:50,884 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 6건째: JJ02_6565SL10 - 필수 Recipe 파일 없음: JJ02_6565SL10/Sensitivity1.xml
2026-10-28 06:01:51,007 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 7건째: JJ02_6965SL10 - 필수 Recipe 파일 없음: JJ02_6965SL10/Sensitivity1.xml
2026-10-28 06:01:51,151 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 8건째: JJ02_7071SL10 - 필수 Recipe 파일 없음: JJ02_7071SL10/Sensitivity1.xml
2026-10-28 06:01:51,326 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 9건째: JJ02_7077SL10 - 필수 Recipe 파일 없음: JJ02_7077SL10/Sensitivity1.xml
2026-10-28 06:01:51,503 WARNING src.collectors.ftp.collector - [SMPW06] Recipe 제외(필수 파일 없음) 10건째: JJ02_7091SL10 - 필수 Recipe 파일 없음: JJ02_7091SL10/Sensitivity1.xml
2026-10-28 06:15:14,095 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 06:35:42,493 WARNING src.collectors.ftp.collector - MES metadata is ambiguous for device=MD12 (2 values)
2026-10-28 06:35:42,605 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 06:55:58,469 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 07:16:37,721 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 07:29:14,465 ERROR src.collectors.ftp.collector - [SMPW06] Recipe download failed: PU04_3365SL10 (1 consecutive failures)
Traceback (most recent call last):
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\collector.py", line 420, in _scan_recipe
    safe_download_recipe(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 272, in safe_download_recipe
    ftp_download_tree(
  File "D:\개발\5_DM_RMS\3_dm_backup\fab1_v3\src\collectors\ftp\client.py", line 221, in ftp_download_tree
    with part.open("wb") as stream:
         ^^^^^^^^^^^^^^^
  File "C:\Users\10007166\AppData\Local\Programs\Python\Python312\Lib\pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PermissionError: [Errno 13] Permission denied: 'D:\\DM_DATA_2\\SMPW06\\.PU04_3365SL10.staging-ac840dc93b534b19bf5f8875bdbf44e0\\Test1.xml.part'
2026-10-28 07:37:34,063 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
2026-10-28 07:57:23,792 INFO src.collectors.ftp.collector - [SMPW06] flushed 999 rows
