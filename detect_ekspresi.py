import argparse
import csv
import os
import platform
import sys
import math
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


EXPRESSIONS = {
    0: {'emoji': '', 'label': 'SENANG', 'color': (0, 255, 0)},
    39: {'emoji': '', 'label': 'BIASA', 'color': (255, 255, 0)},
    67: {'emoji': '', 'label': 'TERKEJUT', 'color': (255, 0, 255)},
}

EXPRESSION_POOL = [
    {'emoji': '', 'label': 'SENANG', 'color': (0, 255, 0)},
    {'emoji': '', 'label': 'MARAH', 'color': (0, 0, 255)},
    {'emoji': '', 'label': 'SEDIH', 'color': (255, 165, 0)},
    {'emoji': '', 'label': 'TERKEJUT', 'color': (255, 0, 255)},
    {'emoji': '', 'label': 'BIASA', 'color': (255, 255, 0)},
]

SENANG_KEYWORDS = ['sehat', 'baik', 'bagus', 'fresh', 'matang', 'ripe', 'good', 'healthy', 'person', 'orang']
MARAH_KEYWORDS = ['rusak', 'busuk', 'jelek', 'buruk', 'sampah', 'bad', 'rotten', 'damage', 'broken']
SEDIH_KEYWORDS = ['sedih', 'layu', 'kering', 'mati', 'dead', 'wilt', 'sad', 'cry']
TERKEJUT_KEYWORDS = ['hape', 'hp', 'phone', 'cell', 'handphone', 'surprised', 'shock']
BIASA_KEYWORDS = ['botol', 'bottle', 'meja', 'chair', 'table', 'normal', 'biasa']


def build_expressions(model_names):
    exp = {}
    used = 0
    for cid, cname in model_names.items():
        name_lower = str(cname).lower()
        if any(kw in name_lower for kw in SENANG_KEYWORDS):
            exp[cid] = EXPRESSION_POOL[0]
        elif any(kw in name_lower for kw in MARAH_KEYWORDS):
            exp[cid] = EXPRESSION_POOL[1]
        elif any(kw in name_lower for kw in SEDIH_KEYWORDS):
            exp[cid] = EXPRESSION_POOL[2]
        elif any(kw in name_lower for kw in TERKEJUT_KEYWORDS):
            exp[cid] = EXPRESSION_POOL[3]
        elif any(kw in name_lower for kw in BIASA_KEYWORDS):
            exp[cid] = EXPRESSION_POOL[4]
        else:
            exp[cid] = EXPRESSION_POOL[used % len(EXPRESSION_POOL)]
            used += 1
    return exp


def draw_face(img, x, y, size, expression):
    cx, cy = x + size // 2, y + size // 2
    r = size // 2
    color = expression['color']

    cv2.circle(img, (cx, cy), r, color, 2)

    eye_off = int(r * 0.3)
    eye_r = max(2, r // 6)
    cv2.circle(img, (cx - eye_off, cy - int(r * 0.2)), eye_r, color, -1)
    cv2.circle(img, (cx + eye_off, cy - int(r * 0.2)), eye_r, color, -1)

    if expression['label'] == 'SENANG':
        cv2.ellipse(img, (cx, cy + int(r * 0.1)), (int(r * 0.4), int(r * 0.25)), 0, 0, 180, color, 2)
    elif expression['label'] == 'MARAH':
        cv2.ellipse(img, (cx, cy + int(r * 0.2)), (int(r * 0.4), int(r * 0.2)), 0, 0, 180, color, 2)
        brow_off = int(r * 0.25)
        cv2.line(img, (cx - eye_off - brow_off, cy - int(r * 0.4)), (cx - eye_off + brow_off, cy - int(r * 0.55)), color, 2)
        cv2.line(img, (cx + eye_off - brow_off, cy - int(r * 0.55)), (cx + eye_off + brow_off, cy - int(r * 0.4)), color, 2)
    elif expression['label'] == 'SEDIH':
        cv2.ellipse(img, (cx, cy + int(r * 0.3)), (int(r * 0.4), int(r * 0.2)), 0, 180, 360, color, 2)
        tear_x, tear_y = cx + eye_off, cy - int(r * 0.2) + eye_r
        cv2.line(img, (tear_x, tear_y), (tear_x, tear_y + int(r * 0.3)), (255, 200, 0), 2)
    elif expression['label'] == 'TERKEJUT':
        cv2.ellipse(img, (cx, cy + int(r * 0.2)), (int(r * 0.3), int(r * 0.3)), 0, 0, 360, color, 2)
        cv2.circle(img, (cx - eye_off, cy - int(r * 0.2)), eye_r + 1, (255, 255, 255), -1)
        cv2.circle(img, (cx + eye_off, cy - int(r * 0.2)), eye_r + 1, (255, 255, 255), -1)
        cv2.circle(img, (cx - eye_off, cy - int(r * 0.2)), eye_r, (0, 0, 0), -1)
        cv2.circle(img, (cx + eye_off, cy - int(r * 0.2)), eye_r, (0, 0, 0), -1)


def draw_mood_overlay(img, mood_label, mood_color, counts, cls_names):
    h, w = img.shape[:2]
    overlay = img.copy()
    box_h = 80
    cv2.rectangle(overlay, (w - 260, 10), (w - 10, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.rectangle(img, (w - 260, 10), (w - 10, 10 + box_h), mood_color, 2)

    mood_emoji = {'SENANG': '', 'SEDIH': '', 'MARAH': '', 'TERKEJUT': '', 'BIASA': ''}
    emoji_char = mood_emoji.get(mood_label, '')
    total = sum(counts.values())
    cv2.putText(img, f"{emoji_char} {mood_label}", (w - 245, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mood_color, 2)
    parts = [f"{cls_names.get(k, k)}={v}" for k, v in sorted(counts.items()) if v > 0]
    cv2.putText(img, ', '.join(parts) if parts else 'kosong', (w - 245, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)


def get_mood(detections, cls_map, expressions):
    total = len(detections)
    counts = {}
    labels = []
    for d in detections:
        c = int(d[5])
        counts[c] = counts.get(c, 0) + 1
        exp = expressions.get(c, EXPRESSION_POOL[4])
        labels.append(exp['label'])

    if total == 0:
        return 'BIASA', (200, 200, 200), counts

    senang = labels.count('SENANG')
    marah = labels.count('MARAH')
    sedih = labels.count('SEDIH')
    terkejut = labels.count('TERKEJUT')

    if senang >= total * 0.5:
        return 'SENANG', (0, 255, 0), counts
    elif marah >= total * 0.5:
        return 'MARAH', (0, 0, 255), counts
    elif terkejut >= total * 0.5:
        return 'TERKEJUT', (255, 0, 255), counts
    elif sedih >= total * 0.5:
        return 'SEDIH', (255, 165, 0), counts
    else:
        return 'SENANG', (0, 255, 0), counts


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',
        source=ROOT / 'data/images',
        data=ROOT / 'data/coco128.yaml',
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        save_txt=False,
        save_csv=False,
        save_conf=False,
        save_crop=False,
        nosave=False,
        classes=[0, 39, 67],
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=3,
        hide_labels=False,
        hide_conf=False,
        half=False,
        dnn=False,
        vid_stride=1,
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    expressions = build_expressions(names)
    imgsz = check_img_size(imgsz, s=stride)

    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        csv_path = save_dir / 'predictions.csv'

        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            det_list = []
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                for row in reversed(det):
                    c = int(row[5])
                    conf = row[4]
                    cls = row[5]
                    xyxy = row[:4]
                    det_list.append(row)
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:
                        c = int(cls)
                        cls_name = names[c]
                        expr = expressions.get(c, EXPRESSION_POOL[0])
                        label = None if hide_labels else f'{expr["emoji"]} {cls_name} {conf:.2f}'

                        annotator.box_label(xyxy, label, color=expr['color'])
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            mood_label, mood_color, counts = get_mood(det_list, names, expressions)
            draw_mood_overlay(im0, mood_label, mood_color, counts, names)

            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device (0) atau cpu (auto: GPU jika ada)')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-csv', action='store_true', help='save results in CSV format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 39, 67], help='filter class: default orang(0) botol(39) hape(67)')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
