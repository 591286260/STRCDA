import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, precision_score, auc, accuracy_score, recall_score, f1_score, matthews_corrcoef, \
    precision_recall_curve


def poltting(test_labels, score, path, fold):
    auc_plt = []
    aupr_plt = []

    auc_areas = []
    aupr_areas = []
    F1s = []
    MCCs = []
    ACCs = []
    Precisions = []
    RECALLs = []

    for i in range(fold):
        auc_inner = []
        aupr_inner = []
        fpr, tpr, threshold = roc_curve(test_labels[i], score[i])

        auc_area = auc(fpr, tpr)
        precision, recall, _thresholds = precision_recall_curve(test_labels[i], score[i])
        aupr_area = auc(recall, precision)
        auc_inner.append(fpr)
        auc_inner.append(tpr)
        auc_inner.append(auc_area)
        aupr_inner.append(precision)
        aupr_inner.append(recall)
        aupr_inner.append(aupr_area)
        auc_plt.append(auc_inner)
        aupr_plt.append(aupr_inner)
        score_lables = [0 if j < 0.5 else 1 for j in score[i]]  ##阈值0.5
        target = mutile_scores(test_labels[i], score_lables)
        # print('auc_area:', "{:.2f}".format(auc_area*100), ' aupr_area:', "{:.2f}".format(aupr_area*100),
        #       ' Precision:', "{:.4f}".format(target[2]), ' RECALL:', "{:.4f}".format(target[3]),
        #       ' F1:', "{:.4f}".format(target[0]), ' ACC:', "{:.4f}".format(target[4]), ' MCC:', "{:.4f}".format(target[1]))


        print("{:.4f}".format(auc_area), "{:.4f}".format(aupr_area), "{:.2f}".format(target[2]*100), "{:.2f}".format(target[3]*100),
              "{:.2f}".format(target[0]*100), "{:.2f}".format(target[4]*100), "{:.2f}".format(target[1]*100))

        auc_areas.append(auc_area)
        aupr_areas.append(aupr_area)
        F1s.append(target[0])
        MCCs.append(target[1])
        Precisions.append(target[2])
        RECALLs.append(target[3])
        ACCs.append(target[4])


    print("{:.4f}±{:.4f}".format(np.mean(auc_areas), np.std(auc_areas)),
          "{:.4f}±{:.4f}".format(np.mean(aupr_areas), np.std(aupr_areas)),
          "{:.2f}±{:.2f}".format(np.mean(Precisions)*100, np.std(Precisions)*100),
          "{:.2f}±{:.2f}".format(np.mean(RECALLs)*100, np.std(RECALLs)*100),
          "{:.2f}±{:.2f}".format(np.mean(F1s)*100, np.std(F1s)*100),
          "{:.2f}±{:.2f}".format(np.mean(ACCs)*100, np.std(ACCs)*100),
          "{:.2f}±{:.2f}".format(np.mean(MCCs)*100, np.std(MCCs)*100))

    # print('auc_area:', "{:.2f}±{:.2f}".format(np.mean(auc_areas)*100, np.std(auc_areas)*100),
    #       ' aupr_area:', "{:.2f}±{:.2f}".format(np.mean(aupr_areas)*100, np.std(aupr_areas)*100),
    #       ' Precision:', "{:.4f}±{:.4f}".format(np.mean(Precisions), np.std(Precisions)),
    #       ' RECALL:', "{:.4f}±{:.4f}".format(np.mean(RECALLs), np.std(RECALLs)),
    #       ' F1:', "{:.4f}±{:.4f}".format(np.mean(F1s), np.std(F1s)),
    #       ' ACC:', "{:.4f}±{:.4f}".format(np.mean(ACCs), np.std(ACCs)),
    #       ' MCC:',"{:.4f}±{:.4f}".format(np.mean(MCCs), np.std(MCCs)))



    pltt(auc_plt, path, fold, 'auc')
    pltt(aupr_plt, path, fold, 'aupr')


def pltt(area, path, fold, plotting_type):
    lw = 1.5
    plt.figure(figsize=(7, 4.4))#8, 5    3.5, 2.2
    mean_fpr=np.linspace(0, 1, 100000)
    tprs=[]
    aucs=[]
    mean_recall = np.linspace(0, 1, 100000)
    Ps=[]
    Rs=[]
    RPs=[]
    #设置坐标精度0.1
    x_major_locator=MultipleLocator(0.1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(x_major_locator)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    if plotting_type == 'auc':
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        for i in range(fold):
            fpr = area[i][0].tolist()
            tpr = area[i][1].tolist()
            fpr.insert(0, 0)
            tpr.insert(0, 0)
            plt.plot(fpr, tpr,
                     lw=lw, label='ROC fold' + str(i + 1) + ' (AUC = %0.4f)' % area[i][2])

            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(area[i][2])

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(
            mean_fpr,
            mean_tpr,
            color="black",
            label=r"Mean ROC (AUC = %0.4f $\pm$ %0.4f)" % (mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")##图例位置
        # plt.savefig(path+'11-fig auc.png')

    else:
        plt.plot([1, 0], [0, 1], color='navy', lw=lw, linestyle='--')
        for i in range(fold):
            plt.plot(area[i][1], area[i][0],
                     lw=lw, label='fold' + str(i + 1) + ' (AUPR = %0.4f)' % area[i][2])

            RPs.append(area[i][2])
            Ps.append(np.interp(mean_recall, area[i][0], area[i][1]))

        # mean_precision/=fold
        # mean_average_precision=sum(mean_average_precision)/len(mean_average_precision)
        mean_precision = np.mean(Ps,axis=0)
        mean_RPs = np.mean(RPs,axis=0)
        std_RPS=np.std(RPs)
        plt.plot(
            mean_precision,
            mean_recall,
            color="black",
            label=r"Mean AP (AUPR = %0.4f)" % (mean_RPs),
            lw=2,
            alpha=0.8,
        )


        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # plt.title('Precision Recall Curve')
        plt.legend(loc="lower left")
        # plt.savefig(path+'11-fig aupr.png')
    # plt.show()



def mutile_scores(test_lables, score_lables):
    f1 = f1_score(test_lables, score_lables)
    mcc = matthews_corrcoef(test_lables, score_lables)
    p = precision_score(test_lables, score_lables)
    recall = recall_score(test_lables, score_lables)
    acc=accuracy_score(test_lables,score_lables)
    target = [f1, mcc, p, recall,acc]
    return target
