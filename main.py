import argparse
import cv2 as cv

def draw_contours_rects(contours: cv.typing.MatLike, min_area: int, frame: cv.typing.MatLike):

    large_contours = [cnt for cnt in contours if cv.contourArea(cnt) > min_area]
    large_rects = [cv.boundingRect(cnt) for cnt in large_contours]
    # large_rects, _ = cv.groupRectangles(large_rects, 1, 1.5)
    frame_out = frame
    for rect in large_rects:
        frame_out = cv.rectangle(frame, rect, (255, 127, 0), 4)

    return frame_out

def camera_function_bkg(args):

    backSub = cv.createBackgroundSubtractorMOG2(500, 50, True)
    # backSub = cv.createBackgroundSubtractorKNN(500, 200, True)

    print("start of camera function: BKG")

    if (args.video == ""):
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(args.video)

    if not cap.isOpened():
        print("canno open the camera")
        return

    imW = 1280
    imH = 720

    while True:
        ret, frame = cap.read()

        if ret:
            if (args.video != ""):
                frame = cv.resize(frame, (imW, imH), interpolation = cv.INTER_AREA)

            fgMask = backSub.apply(frame)

            ret, thresh = cv.threshold(fgMask, args.threshold, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
            contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            # frame_ct = cv.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

            frame_out = draw_contours_rects(contours, args.contour, frame.copy())

            # cv.imshow(">-'(**)'-< crab rave", frame_ct)
            cv.imshow("big biznis here...", frame_out)

            if cv.waitKey(4) & 0xff == ord('q'):
                break
        else:
            print("something went wrong")
            break

    print("camera shutdown")
    cap.release()
    cv.destroyAllWindows()
    return

def camera_function_2gray(args):

    print("start of camera function: 2GRAY")
    if (args.video == ""):
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(args.video)

    if not cap.isOpened():
        print("canno open the camera")
        return

    blur_size = 15
    imW = 1280
    imH = 720

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to capture initial frame.")
        return
    if (args.video != ""):
        prev_frame = cv.resize(prev_frame, (imW, imH), interpolation = cv.INTER_AREA)

    while True:
        ret, frame = cap.read()

        if ret:
            if (args.video != ""):
                frame = cv.resize(frame, (imW, imH), interpolation = cv.INTER_AREA)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (blur_size, blur_size), 0)

            prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
            prev_gray = cv.GaussianBlur(prev_gray, (blur_size, blur_size), 0)

            prev_frame = frame.copy()

            # Compute the absolute difference between the current frame and first_frame
            frame_diff = cv.absdiff(prev_gray, gray)

            # Threshold the difference image
            ret, thresh = cv.threshold(frame_diff, args.threshold, 255, cv.THRESH_BINARY)
            # thresh = cv.adaptiveThreshold(frame_diff, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

            # Dilate the thresholded image to fill in holes
            thresh = cv.dilate(thresh, None, iterations = 2)

            # Find contours
            contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            frame_ct = cv.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)

            frame = draw_contours_rects(contours, args.contour, frame)

            cv.imshow("big biznis here...", frame)
            # cv.imshow("Threshold", thresh)
            # cv.imshow("Blur Gray", gray)
            # cv.imshow(">-'(**)'-< crab rave", frame_ct)

            if cv.waitKey(4) & 0xff == ord('q'):
                break
        else:
            print("something went wrong")
            break

    print("camera shutdown")
    cap.release()
    cv.destroyAllWindows()
    return

def main(args):
    print("let's start and we'll see...")

    if (args.threshold > 255):
        print("threshold value greater than 255 is useless...")
        return

    if (args.algo == "gray"):
        camera_function_2gray(args)
    else:
        if (args.algo == "mask"):
            camera_function_bkg(args)
        else:
            print("wrong argument: algo")
            return

    print("thats's all, folks...")

# . . . . . . entry point . . . . . . . . . . . . . . . . . . . . . . . . . . .
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="moving detecion demo")

    parser.add_argument("--video", default="", help="Video to play. Camera stream is opened, if omitted")
    parser.add_argument("--algo", default="mask", help="Algorithm to use: 'mask' - for background mask subrtraction (default), "
                                                       "'gray' - for color to gray conversion")
    parser.add_argument("--threshold", type=int, default=50, help="detection threshold, default is 50, max is 255")
    parser.add_argument("--contour", type=int, default=200, help="minimum contour size to draw, defauly is 200")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Pass the parsed arguments object to the main function
    main(args)