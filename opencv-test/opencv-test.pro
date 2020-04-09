TEMPLATE = subdirs
LIBS += $(shell pkg-config opencv --libs)
SUBDIRS += \
    images-and-array-types \
    read-image \
    read-video \
    read-webcam
