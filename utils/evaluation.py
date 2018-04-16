def Modulartiy(A, coms, sums):
    Q = 0.0
    for eachc in coms:
        li = 0
        for eachp in coms[eachc]:
            for eachq in coms[eachc]:
                li += A[eachp][eachq]
        li /= 2
        di = 0
        for eachp in coms[eachc]:
            for eachq in range(vertices):
                di += A[eachp][eachq]
        Q = Q + (li - (di * di) /(sums*4))
    Q = Q / float(sums)
    return Q

def JaccardIndex(cluA,cluB):
    a, b, c = 0
    for i in range(vertices):
        for j in range(vertices):
            if cluA[i] == cluA[j] and cluB[i] == cluB[j]:
                a += 1
            if cluA[i] == cluA[j] and cluB[i] == cluB[j]:
                b += 1
            if cluA[i] == cluA[j] and cluB[i] == cluB[j]:
                c += 1
    return a/(double)(a+b+c)

def FsameIndex(cluA,cluB):
    S = mat([[0 for i in range(len(cluB))] for j in range(len(cluA))])
    for i in range(vertices):
        S[cluA[i]][cluB[i]] = 1
    r = sum(S.max(0))
    c = sum(S.max(1))
    fsame = 50.0*(r+c)/float(Vertices)
    return fsame

def NMI(cluA,cluB):
    #混淆矩阵
    cmat = [[0 for i in range(len(cluA))] for j in range(len(cluB))]
    i = 0
    j = 0
    for eacha in cluA:
        for eachb in cluB:
            cmat[i][j] = len(set(cluA[eacha]) & set(cluB[eachb]))
            j += 1
        i += 1
        j = 0
    #print cmat
    #the nmi_numerator part
    nmi_numerator = 0.0
    for i in range(len(cluA)):
        for j in range(len(cluB)):
            if (cmat[i][j]!=0):
                row = 0
                column = 0
                for k in range(len(cluB)):
                    row = row + cmat[i][k]
                for l in range(len(cluA)):
                    column = column + cmat[l][j]
                nmi_numerator = nmi_numerator + cmat[i][j] * log10((cmat[i][j] * vertices)/float(row * column))
    nmi_numerator = -2 * nmi_numerator
    #the denominator part
    nmi_denominator1 = 0.0
    nmi_denominator2 = 0.0
    nmi = 0.0
    for i in range(len(cluA)):
        row = 0
        for k in range(len(cluB)):
            row = row + cmat[i][k]
            if (row != 0):
                nmi_denominator1 = nmi_denominator1 + row * log10(row / float(vertices))
            for j in range(len(cluB)):
                column = 0
                for l in range(len(cluA)):
                    column = column + cmat[l][j];
                if (column != 0):
                    nmi_denominator2 = nmi_denominator2 + column * log10(column / float(vertices))
            nmi_denominator = nmi_denominator1 + nmi_denominator2
            print
            nmi_numerator, nmi_denominator
            if (nmi_denominator != 0):
                nmi = nmi_numerator / float(nmi_denominator)
            return nmi