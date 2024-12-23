


from ml_service.detect_handsign import HandSignDetector


detector = HandSignDetector('saved_model/best2.pt')

detector.detect('image/exx.jpg')


