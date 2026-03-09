#include "Tracker.hpp"

void track( Mat &frame, Ptr<Tracker> tracker, Rect2i &trackingBox ) {
    // Update the tracker with the new frame
    if (tracker->update(frame, trackingBox)) {
        // Draw the tracking box
        rectangle(frame, trackingBox, Scalar(255, 0, 0), 2, 1);
    } else {
        cout << "Tracking failure detected." << endl;
    }
}
