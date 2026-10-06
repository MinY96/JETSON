import argparse, csv
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass
class Config:
    analysis_width:int=640; cst_up_thr:float=.12; stable_thr:float=.055
    stable_window_sec:float=.12; search_start:float=.15; search_end:float=.90
    peak_thr_ratio:float=.20; min_peak_dist_ratio:float=.06
    blade_center_ratio:float=.50; blade_half_ratio:float=.18
    wafer_min_gap_ratio:float=.035; wafer_max_gap_ratio:float=.42
    debug_video:bool=True; debug_scale:float=.75

def resize(img,w):
    h,ow=img.shape[:2]
    if ow<=w:return img.copy(),1.
    s=w/ow
    return cv2.resize(img,(w,max(1,round(h*s))),interpolation=cv2.INTER_AREA),s

def select_roi(frame):
    h,w=frame.shape[:2]; s=min(1280/w,800/h,1.)
    p=cv2.resize(frame,None,fx=s,fy=s) if s<1 else frame
    r=cv2.selectROI("Select measurement ROI",p,False,False); cv2.destroyAllWindows()
    if r[2]<=0 or r[3]<=0: raise RuntimeError("ROI not selected")
    return tuple(round(v/s) for v in r)

def prep(frame,roi,cfg):
    """
    User ROI 중 LEFT 50%만 실제 분석에 사용한다.
    우측의 지속적인 빛번짐을 Optical Flow / Sobel / 판정에서 완전히 제외한다.
    Overlay 좌표 환산을 위해 resize scale을 함께 반환한다.
    """
    x,y,w,h=roi
    analysis_w=max(1,w//2)
    crop=frame[y:y+h,x:x+analysis_w]
    sm,s=resize(crop,cfg.analysis_width)
    g=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(5,5),0)
    return g,s

def flow_y(a,b):
    f=cv2.calcOpticalFlowFarneback(a,b,None,.5,2,21,2,5,1.2,0)
    fx,fy=f[...,0],f[...,1]; mag=cv2.magnitude(fx,fy); m=mag>=.12
    if m.sum()<max(10,int(m.size*.002)): return 0.,0.
    return float(np.median(fy[m])),float(m.mean())

def smooth(v,n):
    n=max(1,n); return np.convolve(np.asarray(v,np.float32),np.ones(n)/n,"same")

def choose_frame(rec,fps,cfg):
    n=len(rec); vy=np.array([r["vy"] for r in rec]); act=np.array([r["activity"] for r in rec])
    sv=smooth(vy,max(1,round(.08*fps))); a=max(1,round(n*cfg.search_start)); b=min(n-2,round(n*cfg.search_end))
    wn=max(2,round(cfg.stable_window_sec*fps)); dn=max(1,round(.06*fps)); best=None
    for sign in (1.,-1.):
        d=sv*sign
        for i in range(a+dn,b-wn):
            pulse=float(np.mean(d[i-dn:i])); stable=float(np.mean(np.abs(sv[i:i+wn])))
            if pulse<cfg.cst_up_thr or stable>cfg.stable_thr: continue
            score=pulse-1.5*stable-.15*float(np.mean(act[i:i+wn]))
            if best is None or score>best["score"]:
                best=dict(score=score,sign=int(sign),pulse=pulse,stable=stable,idx=min(n-1,i+wn//2),fallback=False)
    if best is None:
        pi=a+int(np.argmax(np.abs(sv[a:b]))); lo=min(b-1,pi+1); hi=min(b,pi+max(wn*5,round(.5*fps)))
        cand=[(float(np.mean(np.abs(sv[i:min(n,i+wn)]))),i) for i in range(lo,hi) if min(n,i+wn)-i>=2]
        if not cand:return None,{}
        st,i=min(cand); best=dict(score=-st,sign=0,pulse=float(abs(sv[pi])),stable=st,idx=min(n-1,i+wn//2),fallback=True)
    return best["idx"],best

def profile(gray,cfg):
    sy=np.abs(cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3))
    w=gray.shape[1]

    # 이미 원 ROI의 LEFT 50%만 사용하므로 좌/우 가장자리 3%만 제외한다.
    x0=max(0,int(.03*w))
    x1=min(w,int(.97*w))
    if x1<=x0:
        x0,x1=0,w

    p=np.percentile(sy[:,x0:x1],70,axis=1).astype(np.float32)

    # OpenCV 4.14: kernel dimension에 0을 사용할 수 없다.
    # profile은 세로(Y) 방향 1D signal이므로 1x5 kernel로 smoothing.
    p=cv2.GaussianBlur(
        p.reshape(-1,1),
        (1,5),
        sigmaX=0,
        sigmaY=2.0
    ).ravel()
    return p

def peaks(p,min_dist,thr):
    c=sorted([(float(p[i]),i) for i in range(1,len(p)-1) if p[i]>=p[i-1] and p[i]>p[i+1] and p[i]>=thr],reverse=True)
    out=[]
    for z in c:
        if all(abs(z[1]-q[1])>=min_dist for q in out):out.append(z)
    return sorted(out,key=lambda z:z[1])

def measure(gray,cfg):
    h=gray.shape[0]; p=profile(gray,cfg); thr=float(p.min()+(p.max()-p.min())*cfg.peak_thr_ratio)
    ps=peaks(p,max(2,round(h*cfg.min_peak_dist_ratio)),thr); ey=h*cfg.blade_center_ratio; bh=h*cfg.blade_half_ratio
    bc=[z for z in ps if abs(z[1]-ey)<=bh]
    if not bc:return None,p,ps
    bs,by=max(bc,key=lambda z:z[0]*(1-.35*min(abs(z[1]-ey)/max(bh,1),1)))
    ming=max(2,round(h*cfg.wafer_min_gap_ratio)); maxg=max(ming+1,round(h*cfg.wafer_max_gap_ratio))
    up=[z for z in ps if ming<=by-z[1]<=maxg]; lo=[z for z in ps if ming<=z[1]-by<=maxg]
    if not up or not lo:return None,p,ps
    us,uy=min(up,key=lambda z:(by-z[1],-z[0])); ls,ly=min(lo,key=lambda z:(z[1]-by,-z[0]))
    dt,db=by-uy,ly-by; total=dt+db
    return dict(uy=uy,by=by,ly=ly,dt=dt,db=db,tr=dt/total,br=db/total,confidence=float(np.mean([us,bs,ls])/max(p.max(),1e-6))),p,ps

def overlay(frame,roi,m,s,name,idx,fps):
    o=frame.copy(); x,y,w,h=roi; inv=1/s
    analysis_w=max(1,w//2)
    uy=round(y+m["uy"]*inv); by=round(y+m["by"]*inv); ly=round(y+m["ly"]*inv)

    # Yellow: original requested ROI / Cyan: actual analysis area (LEFT 50%)
    cv2.rectangle(o,(x,y),(x+w,y+h),(0,255,255),1)
    cv2.rectangle(o,(x,y),(x+analysis_w,y+h),(255,255,0),2)
    dt,db=by-uy,ly-by; tot=max(1,dt+db)
    for yy,c,t in [(uy,(255,0,0),"UPPER WAFER"),(by,(0,255,0),"BLADE CENTER"),(ly,(255,0,0),"LOWER WAFER")]:
        cv2.line(o,(x+round(.03*analysis_w),yy),(x+round(.97*analysis_w),yy),c,2); cv2.putText(o,f"{t} Y={yy}",(x+10,max(25,yy-7)),0,.6,c,2)
    cv2.rectangle(o,(20,20),(610,175),(0,0,0),-1)
    vals=[f"TYPE_2 GAP | {name}",f"Frame={idx} Time={idx/fps:.3f}s",f"Top={dt}px ({dt/tot*100:.1f}%)",f"Bottom={db}px ({db/tot*100:.1f}%)",f"Ratio={dt/tot*10:.2f}:{db/tot*10:.2f}"]
    for i,t in enumerate(vals):cv2.putText(o,t,(35,48+i*28),0,.58,(255,255,255),2)
    return o,dict(upper_y=uy,blade_center_y=by,lower_y=ly,top_gap_px=dt,bottom_gap_px=db,top_ratio=dt/tot,bottom_ratio=db/tot)

def save_timing_debug(d, frames, rec, mi, fps, fw, fh, roi, cfg):
    """Timing diagnostics are always saved, independent of line detection."""
    with (d/"timing_debug.csv").open("w",newline="",encoding="utf-8-sig") as csv_fh:
        fields=["frame_idx","time_sec","vy","activity","is_measurement_frame"]
        w=csv.DictWriter(csv_fh,fieldnames=fields)
        w.writeheader()
        for r in rec:
            row=dict(r)
            row["time_sec"]=row["frame_idx"]/fps
            row["is_measurement_frame"]=int(mi is not None and row["frame_idx"]==mi)
            w.writerow(row)

    if cfg.debug_video:
        dw=max(1,round(fw*cfg.debug_scale))
        dh=max(1,round(fh*cfg.debug_scale))
        vw=cv2.VideoWriter(
            str(d/"timing_debug.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (dw,dh)
        )
        for j,fr in enumerate(frames):
            z=fr.copy()
            x,y,rw,rh=roi
            cv2.rectangle(z,(x,y),(x+rw,y+rh),(0,255,255),1)
            c=(0,255,0) if mi is not None and j==mi else (255,255,0)
            cv2.rectangle(z,(x,y),(x+rw//2,y+rh),c,2)
            cv2.putText(
                z,
                f"frame={j} time={j/fps:.3f}s VY={rec[j]['vy']:+.4f} ACT={rec[j]['activity']:.4f}",
                (25,40),0,.65,(255,255,255),2
            )
            if mi is not None and j==mi:
                cv2.putText(z,"MEASUREMENT FRAME",(25,75),0,.8,(0,255,0),2)
            vw.write(cv2.resize(z,(dw,dh)))
        vw.release()


def save_line_debug(d, frame, gray, p, ps, roi, scale, cfg):
    """Save profile values and every detected Y-peak even when final 3-line matching fails."""
    h=gray.shape[0]
    thr=float(p.min()+(p.max()-p.min())*cfg.peak_thr_ratio) if len(p) else 0.0
    peak_map={int(y):float(score) for score,y in ps}

    with (d/"line_profile.csv").open("w",newline="",encoding="utf-8-sig") as fh:
        fields=["y_analysis","profile_value","is_peak","peak_score","threshold"]
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for yy,val in enumerate(p):
            w.writerow(dict(
                y_analysis=yy,
                profile_value=float(val),
                is_peak=int(yy in peak_map),
                peak_score=peak_map.get(yy,""),
                threshold=thr,
            ))

    # Actual CCTV frame + all candidate peaks
    o=frame.copy()
    x,y,rw,rh=roi
    analysis_w=max(1,rw//2)
    inv=1.0/max(scale,1e-9)

    cv2.rectangle(o,(x,y),(x+rw,y+rh),(0,255,255),1)
    cv2.rectangle(o,(x,y),(x+analysis_w,y+rh),(255,255,0),2)

    for rank,(score,py) in enumerate(sorted(ps,reverse=True),1):
        oy=round(y+py*inv)
        cv2.line(o,(x,oy),(x+analysis_w,oy),(0,0,255),1)
        cv2.putText(
            o,f"P{rank} y={py} score={score:.1f}",
            (x+8,max(18,oy-3)),0,.42,(0,0,255),1
        )

    cv2.putText(
        o,
        f"LINE DEBUG | peaks={len(ps)} thr={thr:.2f}",
        (25,35),0,.7,(255,255,255),2
    )
    cv2.imwrite(str(d/"line_debug.jpg"),o)

    # Analysis crop visualization with the 1D profile drawn on the right.
    vis=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    if len(p):
        pn=(p-p.min())/max(float(p.max()-p.min()),1e-6)
        pw=max(120,vis.shape[1]//3)
        panel=np.zeros((vis.shape[0],pw,3),np.uint8)
        for yy,v in enumerate(pn):
            xx=int(v*(pw-15))
            cv2.line(panel,(0,yy),(xx,yy),(180,180,180),1)
        for score,py in ps:
            cv2.line(panel,(0,py),(pw-1,py),(0,0,255),1)
        vis=np.hstack([vis,panel])
    cv2.imwrite(str(d/"line_profile_debug.jpg"),vis)


def analyze(path,out,roi,cfg):
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS)
    fw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps<=0:
        fps=30.0

    ok,first=cap.read()
    if not ok:
        raise RuntimeError("cannot read")

    g,s=prep(first,roi,cfg)
    frames=[first]
    rec=[dict(frame_idx=0,vy=0.,activity=0.)]
    prev=g
    i=1

    while True:
        ok,f=cap.read()
        if not ok:
            break
        g,_=prep(f,roi,cfg)
        vy,a=flow_y(prev,g)
        rec.append(dict(frame_idx=i,vy=vy,activity=a))
        frames.append(f)
        prev=g
        i+=1
    cap.release()

    d=out/path.stem
    d.mkdir(parents=True,exist_ok=True)

    mi,timing=choose_frame(rec,fps,cfg)

    # IMPORTANT: save timing diagnostics before any line-detection return.
    save_timing_debug(d,frames,rec,mi,fps,fw,fh,roi,cfg)

    base=dict(
        clip=path.name,
        analysis_roi_x=roi[0],
        analysis_roi_y=roi[1],
        analysis_roi_w=roi[2]//2,
        analysis_roi_h=roi[3],
    )

    if mi is None:
        return dict(**base,status="MEASURE_FRAME_NOT_FOUND")

    f=frames[mi]
    cv2.imwrite(str(d/"measurement_frame.jpg"),f)

    # Save timing selection metadata even if line detection fails.
    base.update(dict(
        measurement_frame=mi,
        measurement_time_sec=mi/fps,
        timing_score=timing.get("score",""),
        timing_fallback=timing.get("fallback",""),
        cst_motion_sign=timing.get("sign",""),
        cst_pulse_strength=timing.get("pulse",""),
        stable_motion=timing.get("stable",""),
    ))

    g,s=prep(f,roi,cfg)
    m,p,ps=measure(g,cfg)

    # IMPORTANT: line diagnostics are also always saved.
    save_line_debug(d,f,g,p,ps,roi,s,cfg)

    if m is None:
        base.update(dict(
            status="LINE_DETECTION_FAILED",
            line_peak_count=len(ps),
            line_profile_max=float(p.max()) if len(p) else "",
            line_profile_min=float(p.min()) if len(p) else "",
            line_threshold=float(p.min()+(p.max()-p.min())*cfg.peak_thr_ratio) if len(p) else "",
        ))
        return base

    ov,v=overlay(f,roi,m,s,path.name,mi,fps)
    op=d/"measurement_overlay.jpg"
    cv2.imwrite(str(op),ov)

    base.update(dict(
        status="OK",
        **v,
        top_ratio_pct=v["top_ratio"]*100,
        bottom_ratio_pct=v["bottom_ratio"]*100,
        ratio_text=f'{v["top_ratio"]*10:.2f}:{v["bottom_ratio"]*10:.2f}',
        line_confidence=m["confidence"],
        line_peak_count=len(ps),
        line_profile_max=float(p.max()) if len(p) else "",
        line_profile_min=float(p.min()) if len(p) else "",
        line_threshold=float(p.min()+(p.max()-p.min())*cfg.peak_thr_ratio) if len(p) else "",
        overlay_path=str(op),
    ))
    return base

def main():
    p=argparse.ArgumentParser();p.add_argument("--input-dir",required=True);p.add_argument("--output-dir",default="./type2_gap_result")
    p.add_argument("--roi",nargs=4,type=int,default=None);p.add_argument("--analysis-width",type=int,default=640);p.add_argument("--cst-up-thr",type=float,default=.12);p.add_argument("--stable-motion-thr",type=float,default=.055)
    p.add_argument("--stable-window-sec",type=float,default=.12);p.add_argument("--blade-band-center-ratio",type=float,default=.50);p.add_argument("--blade-band-half-ratio",type=float,default=.18)
    p.add_argument("--peak-thr-ratio",type=float,default=.20);p.add_argument("--debug-scale",type=float,default=.75);p.add_argument("--no-debug-video",action="store_true")
    a=p.parse_args(); inp=Path(a.input_dir);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); vids=sorted(inp.glob("*.mp4"))
    if not vids:raise FileNotFoundError(inp)
    cfg=Config(analysis_width=a.analysis_width,cst_up_thr=a.cst_up_thr,stable_thr=a.stable_motion_thr,stable_window_sec=a.stable_window_sec,blade_center_ratio=a.blade_band_center_ratio,blade_half_ratio=a.blade_band_half_ratio,peak_thr_ratio=a.peak_thr_ratio,debug_video=not a.no_debug_video,debug_scale=a.debug_scale)
    roi=tuple(a.roi) if a.roi else None
    if roi is None:
        c=cv2.VideoCapture(str(vids[0]));ok,f=c.read();c.release()
        if not ok:raise RuntimeError("cannot read first clip")
        roi=select_roi(f);print("Selected ROI:",roi)
    print("=== Type2 Wafer Gap POC v3: diagnostics-first ===")
    print(f"Requested ROI: {roi}")
    print(f"Actual analysis area (LEFT 50%): {(roi[0], roi[1], roi[2]//2, roi[3])}")
    rows=[]
    for i,v in enumerate(vids,1):
        print(f"[{i}/{len(vids)}] {v.name}")
        try:r=analyze(v,out,roi,cfg)
        except Exception as e:r=dict(clip=v.name,status=f"ERROR: {type(e).__name__}: {e}")
        rows.append(r);print(" ->",r["status"],r.get("ratio_text",""))
    fields=["clip","status","analysis_roi_x","analysis_roi_y","analysis_roi_w","analysis_roi_h","measurement_frame","measurement_time_sec","timing_score","timing_fallback","cst_motion_sign","cst_pulse_strength","stable_motion","line_peak_count","line_profile_min","line_profile_max","line_threshold","upper_y","blade_center_y","lower_y","top_gap_px","bottom_gap_px","top_ratio","bottom_ratio","top_ratio_pct","bottom_ratio_pct","ratio_text","line_confidence","overlay_path"]
    with (out/"type2_wafer_gap_measurements.csv").open("w",newline="",encoding="utf-8-sig") as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    print(f"Complete: {sum(r['status']=='OK' for r in rows)}/{len(rows)} measured")

if __name__=="__main__":main()
