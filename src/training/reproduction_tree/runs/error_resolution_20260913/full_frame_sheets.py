"""Render every raw frame for named Training audit cases; no predictions or labels changed."""
import argparse, hashlib, json, math, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

R = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('cases', nargs='+')
p.add_argument('--modalities', nargs='+', default=['IR', 'Thermal'])
a = p.parse_args()
out = R / 'audit' / 'full_frames'
out.mkdir(exist_ok=True)
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 14)
for case in a.cases:
    meta = json.loads((R/'audit'/'images'/f'{case}.json').read_text(encoding='utf-8'))
    records = []
    for mod in a.modalities:
        candidates = [Path(r['path']) for r in meta['source_images'] if mod in Path(r['path']).parts]
        if not candidates:
            continue
        folder = candidates[0].parent
        assert 'Training' in folder.parts and 'Test' not in folder.parts
        frames = sorted(x for x in folder.iterdir() if x.suffix.lower() in ['.jpg','.png'])
        for page, start in enumerate(range(0, len(frames), 20)):
            subset = frames[start:start+20]
            canvas = Image.new('RGB', (1280, 48+230*math.ceil(len(subset)/4)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8,5), f'{case} {meta["clip_id"]} | {mod} ALL frames {start}-{start+len(subset)-1}/{len(frames)}', font=font, fill='black')
            rec = {'case_id':case,'modality':mod,'page':page,'total_frames':len(frames),'frames':[]}
            for j, src in enumerate(subset):
                im = Image.open(src).convert('RGB')
                im = ImageOps.contain(im, (320,200))
                x,y = (j%4)*320,48+(j//4)*230
                canvas.paste(im,(x,y))
                draw.text((x+2,y+201), f'#{start+j} {src.stem[-30:]}', font=font, fill='black')
                rec['frames'].append({'index':start+j,'path':str(src),'sha256':hashlib.sha256(src.read_bytes()).hexdigest()})
            dest = out/f'{case}_{mod}_{page:02d}.jpg'
            canvas.save(dest, quality=90)
            rec.update(image=str(dest),image_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),visually_reviewed=False)
            dest.with_suffix('.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            records.append(str(dest))
    print(json.dumps({'case_id':case,'sheets':records,'at_unix':time.time()}),flush=True)
