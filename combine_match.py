import feature_match
import hist_match
import logging


def main(target, compare):
    featureout = feature_match.main(target, compare)
    histout = hist_match.main(target, compare)
    combineout ={}
    for feature in featureout:
        featurevalue = featureout[feature]
        histvalue = histout[feature]
        histvalue *= -1
        histvalue += 1
        combinevalue = histvalue*featurevalue
        #print(str(feature), " Value:", combinevalue)
        combineout[feature] = combinevalue
    for k, v in sorted(combineout.items(), reverse=False, key=lambda x: x[1]):
        logging.info('%s: %f.' % (k, v))

main("picture/roadstar2.jpeg", "picture/")