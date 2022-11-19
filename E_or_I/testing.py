import persondetect as ps

ps.run(weights='yolov5s.pt',source='data/nmask/images/train', save_txt=True, project='result', name='mbti')
