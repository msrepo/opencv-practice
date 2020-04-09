TEMPLATE = subdirs
LIBS += $(shell pkg-config opencv --libs)
SUBDIRS += \
    filtering-and-convolution \
    images-and-array-types \
    read-image \
    read-video \
    read-webcam
