# Motivation

Modern time-domain surveys monitor large swaths of the sky to look for interesting astronomical objects, including near-earth objects, 
unidentified planetesimals, and transits. Imperfect CCD, optics reflection, atmospheric effects and a bunch of other factors contribute 
to the presence of bogus detections. They contaminates real detections of transitions, Variable stars and planetesimals, 
adding to the difficulty against our effort to spot real interesting celestial objects.

Taking into account that we expect to observe ~200k objects in one single exposure in one of the four apertures on Pan-STARRS telescope,
it is impossible for us to manually examine whether a detection is real or bogus. Also, sometimes visual examination helps, 
but does not solve the problem either, due to either huma error in such big volumes of data, and the cost of time. 
We need an automatic solution to distinguish bogus detections from real ones.

Full report including code can be seen in Real-Bogus-Classifiers.ipynb, and a presentation is also available [here](https://zizhengxu.github.io/my_portfolio/Real-BogusClassifier/Real-Bogus-Classifiers.pptx).
