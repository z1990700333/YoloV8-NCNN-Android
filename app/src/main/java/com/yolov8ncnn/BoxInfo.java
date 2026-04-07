package com.yolov8ncnn;

/**
 * Detection box data class.
 */
public class BoxInfo {
    public float x1, y1, x2, y2;
    public float score;
    public int label;

    public BoxInfo(float x1, float y1, float x2, float y2, float score, int label) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.score = score;
        this.label = label;
    }

    public float centerX() {
        return (x1 + x2) / 2.0f;
    }

    public float centerY() {
        return (y1 + y2) / 2.0f;
    }

    public float width() {
        return x2 - x1;
    }

    public float height() {
        return y2 - y1;
    }

    @Override
    public String toString() {
        return String.format("Box[label=%d, score=%.2f, (%.0f,%.0f)-(%.0f,%.0f)]",
                label, score, x1, y1, x2, y2);
    }
}
