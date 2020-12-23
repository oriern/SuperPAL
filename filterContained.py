


def checkPairContained(containedCandidateOffset_list, containOffset_list):
    containedList = []
    for containedCandidate in containedCandidateOffset_list:
        contained = False
        for offset in containOffset_list:
            contained_start, contained_end = containedCandidate
            start, end = offset
            if contained_start >= start and contained_end <= end:
                contained = True
        containedList.append(contained)

    notContained = not(all(containedList))  #if all spans are contained
    return notContained



def checkContained(scuOffsetDict,sentenceText, sentenceOffset = 0):
    notContainedDict = {}
    for containedCandidate, containedCandidateOffset_list in scuOffsetDict.items():
        notContainedList = []
        for contain, containOffset_list in scuOffsetDict.items():
            if contain == containedCandidate:
                continue

                #if one of scus is the full sentence, don't filter the other scus.
            full_sent_scu = True if containOffset_list[0][0] - sentenceOffset == 0 and\
                    containOffset_list[0][1] - sentenceOffset > 0.95*(len(sentenceText) - 1) else False
            if full_sent_scu:
                continue
            notContained = checkPairContained(containedCandidateOffset_list, containOffset_list)
            notContainedList.append(notContained)
            # if not notContained:
            #     print(containedCandidate)
            #     print (contain)

        notContainedDict[containedCandidate] = all(notContainedList)

    return notContainedDict


