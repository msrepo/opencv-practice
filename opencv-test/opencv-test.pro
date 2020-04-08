TEMPLATE = subdirs
LIBS += $(shell pkg-config opencv --libs)
SUBDIRS += \
    ../read-image \
    read-video
