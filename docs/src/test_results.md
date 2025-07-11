# Test Results

Each algorithm is tested upon the same minimal Data: a Sinvewave and a Squarewave mixed together.
![Mixed Testdata](media/jumbled.svg) \

## JADE
The Jade algorithm has no issues recovering the original sources:
![Jade Recovery](media/jade_out.svg) \

## SHIBBS
Shibbs, as an iterative algorithm, needs multiple runs to find the original sources:
![Shibbs Recovery](media/shibbs_out.svg) \

## PICARD
The Picard algorithm, like Jade, recovers the original sources faithfully:
![Picard Recovery](media/picard_out.svg)