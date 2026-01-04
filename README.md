# musical_predictions
Based on my blogposts about the relationship between music and prediction. And we will test the idea of human pattern recognition with that of CNN and Time-Series analysis based machine learning model for trading.

Here's the gist of things:

Time-Series Analysis (Detects volatility compression)
                        |
                        V
Convolution Neual Network (Direction of breakout)



TSA (Time-Series Analysis from now on) is going to give us states, whether it is tradeable right now or not, because at states where we are practically trying to line up the CNN model onto chop and noise is going to give us bad trades and take up a lot of computing power. Our CNN model will be multi dimensional, so we can save some computing power for just math for the TSA. 

A dedicated TSA system for this project is much more suitable than putting together a bunch of indicators because we can look for specific states rather than a "we broke average by some standard deviation". In musical terms, its like hunting for V chords vs giving the whole piece context and "feeling" for the V function, we don't have to know how it will cadence yet, we just need to know we are surely in the V function. 

CNN's job is now very simple, we can have TSA give us more metrics than simply a binary state of "trade" or "no trade". CNN can take the past data and build vast library of different types of "cadences". The different templates of candences will be stored to be utilized for pattern recognition in deployment. This is much more accurate than relying on TSA for the direction of the breakout since all tickers behave differently, similar to how different composers have different nuiances, the different groups of people that trade certain tickers have its own behaviors that cannont be contained in simple labels, same with how we can't say "every composer goes from V to I in a candece" yes we know, but the exact notes and voicings used will be different per each piece and composer, but we can utilize past patterns to figure out what might happen in the current cadence.



My CSV protocol for this project:

raw data (OHLV)
date,time,open,high,low,close,volume

trained parameters
(tbd...)
