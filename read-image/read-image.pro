TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
LIBS += $(shell pkg-config opencv --libs)
SOURCES += \
        main.cpp
