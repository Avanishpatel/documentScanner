import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import java.io.File;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class DocumentScannerImpl {


    private IplImage downScaleImage(IplImage srcImage, int percent) {

        System.out.println("Src Image Height : " + srcImage.height() + " Width : " + srcImage.width());

        IplImage destImage = cvCreateImage(cvSize((srcImage.width() * percent) / 100, (srcImage.height() * percent) / 100), srcImage.depth(), srcImage.nChannels());

        cvResize(srcImage, destImage);

        System.out.println("Dest Image Height : " + destImage.height() + " Width : " + destImage.width());
        return destImage;
    }


    private IplImage applyCannySquareEdgeDetectionOnImage(IplImage srcImage, int percent) {

        IplImage destImage = downScaleImage(srcImage, percent);
        IplImage grayImage = cvCreateImage(cvGetSize(destImage), IPL_DEPTH_8U, 1);

        // convert to gray
        cvCvtColor(destImage, grayImage, CV_BGR2GRAY);
        OpenCVFrameConverter.ToMat convertToMat = new OpenCVFrameConverter.ToMat();
        Frame grayImageFrame = convertToMat.convert(grayImage);
        Mat grayImageMat = convertToMat.convert(grayImageFrame);

        // apply guassian blur
        GaussianBlur(grayImageMat, grayImageMat, new Size(5, 5), 0.0, 0.0, BORDER_DEFAULT);

        destImage = convertToMat.convertToIplImage(grayImageFrame);

        // clean it for better detection
        cvErode(destImage, destImage);
        cvDilate(destImage, destImage);

        // apply canny edge detection
        cvCanny(destImage, destImage, 75.0, 200.0);

        // save image with canny edge detection
        File f = new File(System.getProperty("user.home") + File.separator + "/Downloads/documentscanner/doc-canny-detect.jpeg");
        cvSaveImage(f.getAbsolutePath(), destImage);

        return destImage;
    }


    private CvSeq findLargestSquareOnCannyDetectedImage(IplImage cannyEdgeDetectedImage) {


        IplImage foundedCountoursImage = cvCloneImage(cannyEdgeDetectedImage);

        CvMemStorage memory = CvMemStorage.create();
        CvSeq countours = new CvSeq();

        cvFindContours(foundedCountoursImage, memory, countours, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

        int maxWidth = 0;
        int mAXHeight = 0;

        CvRect countour = null;
        CvSeq seqFounded = null;
        CvSeq nextSeq = new CvSeq();

        for (nextSeq = countours; nextSeq != null; nextSeq = nextSeq.h_next()) {

            countour = cvBoundingRect(nextSeq, 0);
            if ((countour.width() >= maxWidth) && (countour.height() >= mAXHeight)) {

                maxWidth = countour.width();
                mAXHeight = countour.height();
                seqFounded = nextSeq;
            }
        }

        CvSeq result = cvApproxPoly(seqFounded, Loader.sizeof(CvContour.class), memory, CV_POLY_APPROX_DP, cvContourPerimeter(seqFounded) * 0.02, 0);
        for (int i = 0; i < result.total(); i++) {
            CvPoint v = new CvPoint(cvGetSeqElem(result, i));
            cvDrawCircle(foundedCountoursImage, v, 5, CvScalar.BLUE, 20, 8, 0);
            System.out.println("found point(" + v.x() + "," + v.y() + ")");
        }
        File f = new File(System.getProperty("user.home") + File.separator + "/Downloads/documentscanner/doc-find-contours.jpeg");
        cvSaveImage(f.getAbsolutePath(), foundedCountoursImage);
        return result;
    }


    private void applyPersepectiveTransformationThresoldOnOriginalImage(IplImage srcImage, CvSeq contour, int percentage) {

        IplImage warpImage = cvCloneImage(srcImage);

        // first, adjust the image
        for (int i = 0; i < contour.total(); i++) {

            CvPoint point = new CvPoint(cvGetSeqElem(contour, i));

            point.x((point.x() * 100) / percentage);
            point.y((point.y() * 100) / percentage);
        }

        // get each corner point of the image

        CvPoint topRigthPoint = new CvPoint(cvGetSeqElem(contour, 0));
        CvPoint topLeftPoint = new CvPoint(cvGetSeqElem(contour, 1));
        CvPoint bottomLeftPoint = new CvPoint(cvGetSeqElem(contour, 2));
        CvPoint bottomRigthPoint = new CvPoint(cvGetSeqElem(contour, 3));


        int resultWidth = (topRigthPoint.x() - topLeftPoint.x());
        int bottomWidth = (bottomRigthPoint.x() - bottomLeftPoint.x());

        if (bottomWidth > resultWidth) {
            resultWidth = bottomWidth;
        }

        int resultHeight = (bottomLeftPoint.y() - topLeftPoint.y());
        int bottomHeight = (bottomRigthPoint.y() - topRigthPoint.y());

        if (bottomHeight > resultHeight) {
            resultHeight = bottomHeight;
        }

        float[] sourcePoints = {
                topLeftPoint.x(),
                topLeftPoint.y(),
                topRigthPoint.x(),
                topRigthPoint.y(),
                bottomLeftPoint.x(),
                bottomLeftPoint.y(),
                bottomRigthPoint.x(),
                bottomRigthPoint.y()
        };


        float[] destinationPoints = {
                0, 0, resultWidth, 0, 0, resultHeight, resultWidth, resultHeight
        };

        CvMat homegraphy = cvCreateMat(3, 3, CV_32FC1);

        cvGetPerspectiveTransform(sourcePoints, destinationPoints, homegraphy);
        System.out.println(homegraphy.toString());

        IplImage destImage = cvCloneImage(warpImage);
        cvWarpPerspective(warpImage, destImage, homegraphy, CV_INTER_LINEAR, CvScalar.ZERO);


        cropImage(destImage, 0, 0, resultWidth, resultHeight);
    }

    private IplImage cropImage(IplImage srcImage, int fromX, int fromY, int toWidth, int toHeight) {


        cvSetImageROI(srcImage, cvRect(fromX, fromY, toWidth, toHeight));

        IplImage destImage = cvCloneImage(srcImage);
        cvCopy(srcImage, destImage);

        File f = new File(System.getProperty("user.home") + File.separator + "/Downloads/documentscanner/doc-cropped.jpeg");
        cvSaveImage(f.getAbsolutePath(), destImage);
        return destImage;
    }


    public void cleanImage(final String documentFileImagePath) {

        final File documentImageFile = new File(documentFileImagePath);

        final String documentPath = documentImageFile.getAbsolutePath();
        System.out.println(documentPath);

        IplImage docImage = cvLoadImage(documentPath);
        IplImage cannyEdgeImage = applyCannySquareEdgeDetectionOnImage(docImage, 100);

        CvSeq largestSquare = findLargestSquareOnCannyDetectedImage(cannyEdgeImage);

        applyPersepectiveTransformationThresoldOnOriginalImage(docImage, largestSquare, 100);

    }

}
