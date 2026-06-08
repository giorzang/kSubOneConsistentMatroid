CXX      = g++
CPPFLAGS = -std=c++17 -O2 -Wall -Isrc

OMPFLAGS = -fopenmp

LDLIBS   = -lpthread

KFUNC_DEF = -DKFUNC_MAXKCUT

.PHONY: all clean debug

all: maxcut revenue kic sensor preproc preprockic preprosensor

maxcut: src/main.cpp
	$(CXX) src/main.cpp -o maxcut \
	    $(CPPFLAGS) $(OMPFLAGS) -DKFUNC_MAXKCUT $(LDLIBS)

sensor: src/main.cpp
	$(CXX) src/main.cpp -o sensor \
	    $(CPPFLAGS) -DKFUNC_SENSOR_ENTROPY $(LDLIBS)

revenue: src/main.cpp
	$(CXX) src/main.cpp -o revenue \
	    $(CPPFLAGS) -DKFUNC_REVENUE $(LDLIBS)

kic: src/main.cpp
	$(CXX) src/main.cpp -o kic \
	    $(CPPFLAGS) $(OMPFLAGS) -DKFUNC_KIC $(LDLIBS)

preproc: src/data/preprocess.cpp
	$(CXX) src/data/preprocess.cpp -o preproc \
	    $(CPPFLAGS) $(LDLIBS)

preprockic: src/data/preprocess_kic.cpp
	$(CXX) src/data/preprocess_kic.cpp -o preprockic \
	    $(CPPFLAGS) $(LDLIBS)

preprosensor: src/data/preprocess_sensor.cpp
	$(CXX) src/data/preprocess_sensor.cpp -o preprosensor \
	    $(CPPFLAGS) $(LDLIBS)

debug: src/main.cpp
	$(CXX) src/main.cpp -o maxcut_debug \
	    -std=c++17 -O0 -g -ggdb3 -Wall -Isrc -DKFUNC_MAXKCUT $(LDLIBS)

clean:
	rm -f maxcut sensor revenue kic preproc preprockic preprosensor maxcut_debug
