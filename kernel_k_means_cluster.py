import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


test_verileri_1 = np.loadtxt("test_verileri_1.txt", delimiter=" ")
test_verileri_2 = np.loadtxt("test_verileri_2.txt", delimiter=" ")

k = 2 #kernel sayısı
var = 5 #RBF kernel
iteration_counter = 0
input = test_verileri_2
initMethod = "byOriginDistance" #uzaklık methodu

def baslat(data_input, cluster_sayisi, method):
    liste_cluster_uye = [[] for i in range(cluster_sayisi)]
    if (method == "random"):
        shuffled_data = data_input
        np.random.shuffle(shuffled_data)
        for i in range(0, data_input.shape[0]):
            liste_cluster_uye[i%cluster_sayisi].append(data_input[i,:])
    if (method == "byCenterDistance"):
        merkez = np.matrix(np.mean(data_input, axis=0))
        merkez_tekar = np.repeat(merkez, data_input.shape[0], axis=0)
        delta_matris = abs(np.subtract(data_input, merkez_tekar))
        oklit_matris = np.sqrt(np.square(delta_matris).sum(axis=1))
        veri_yeni = np.array(np.concatenate((oklit_matris, data_input), axis=1))
        veri_yeni = veri_yeni[np.argsort(veri_yeni[:, 0])]
        veri_yeni = np.delete(veri_yeni, 0, 1)
        bolum = data_input.shape[0]/cluster_sayisi
        for i in range(0, data_input.shape[0]):
            liste_cluster_uye[np.int64(np.floor(i/bolum))].append(veri_yeni[i,:])
    if (method == "byOriginDistance"):
        origin = np.matrix([[0,0]])
        merkez_tekar = np.repeat(origin, data_input.shape[0], axis=0)
        delta_matris = abs(np.subtract(data_input, merkez_tekar))
        oklit_matris = np.sqrt(np.square(delta_matris).sum(axis=1))
        veri_yeni = np.array(np.concatenate((oklit_matris, data_input), axis=1))
        veri_yeni = veri_yeni[np.argsort(veri_yeni[:, 0])]
        veri_yeni = np.delete(veri_yeni, 0, 1)
        bolum = data_input.shape[0]/cluster_sayisi
        for i in range(0, data_input.shape[0]):
            liste_cluster_uye[np.int64(np.floor(i/bolum))].append(veri_yeni[i,:])

    return liste_cluster_uye

def rbfKernel(data1, data2, sigma):
    delta = abs(np.subtract(data1, data2))
    oklid_kare = (np.square(delta).sum(axis=1))
    sonuc = np.exp(-(oklid_kare)/(2*sigma**2))
    return sonuc

def thirdTerm(uye_cluster):
    sonuc = 0
    for i in range(0, uye_cluster.shape[0]):
        for j in range(0, uye_cluster.shape[0]):
            sonuc = sonuc + rbfKernel(uye_cluster[i, :], uye_cluster[j, :], var)
    sonuc = sonuc / (uye_cluster.shape[0] ** 2)
    return sonuc

def secondTerm(data_input, uye_cluster):
    sonuc = 0
    for i in range(0, uye_cluster.shape[0]):
        sonuc = sonuc + rbfKernel(data_input, uye_cluster[i,:], var)
    sonuc = 2 * sonuc / uye_cluster.shape[0]
    return sonuc

def plot(liste_cluster_uye, centroid, converged):
    n = liste_cluster_uye.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("sonuc")
    plt.clf()
    plt.title("Kernel K-Means")
    for i in range(n):
        col = next(color)
        uye_cluster = np.asmatrix(liste_cluster_uye[i])
        plt.scatter(np.ravel(uye_cluster[:, 0]), np.ravel(uye_cluster[:, 1]), marker=".", s=100, c=col)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]), marker="*", s=400, c=col, edgecolors="black")
    if (converged == 0):
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if (converged == 1):
        plt.show(block=True)

def kernelKMeans(data, initMethod):
    global iteration_counter
    uye = baslat(data, k, initMethod)
    cluster_sayisi = uye.__len__()
    #merkez noktalari son konumlarina ulasana kadar dongu devam ediyor
    while(True):
        #goruntu amacli
        centroid = np.ndarray(shape=(0, data.shape[1]))
        for i in range(0, cluster_sayisi):
            uye_cluster = np.asmatrix(uye[i])
            centroid_cluster = uye_cluster.mean(axis=0)
            centroid = np.concatenate((centroid, centroid_cluster), axis=0)
        kernel_sonuc = np.ndarray(shape=(data.shape[0], 0))
        #verinin rengini ait oldugu cluster'in rengine esitliyoruz
        for i in range(0, cluster_sayisi):
            term3 = thirdTerm(np.asmatrix(uye[i]))
            matris_term3 = np.repeat(term3, data.shape[0], axis=0); matris_term3 = np.asmatrix(matris_term3)
            matris_term2 = np.ndarray(shape=(0,1))
            for j in range(0, data.shape[0]):
                term2 = secondTerm(data[j,:], np.asmatrix(uye[i]))
                matris_term2 = np.concatenate((matris_term2, term2), axis=0)
            matris_term2 = np.asmatrix(matris_term2)
            kernel_sonuc_cluster = np.add(-1*matris_term2, matris_term3)
            kernel_sonuc =\
                np.concatenate((kernel_sonuc, kernel_sonuc_cluster), axis=1)
        cluster_matris = np.ravel(np.argmin(np.matrix(kernel_sonuc), axis=1))
        liste_cluster_uye = [[] for l in range(k)]
        for i in range(0, data.shape[0]):
            liste_cluster_uye[cluster_matris[i].item()].append(data[i,:])
       	
        bool_a = True
        for m in range(0, cluster_sayisi):
            onceki = np.asmatrix(uye[m])
            mevcut = np.asmatrix(liste_cluster_uye[m])
            if (onceki.shape[0] != mevcut.shape[0]):
                bool_a = False
                break
            if (onceki.shape[0] == mevcut.shape[0]):
                bool_cluster = (onceki == mevcut).all()
            bool_a = bool_a and bool_cluster
            if(bool_a == False):
                break
        if(bool_a == True):
            break
        iteration_counter += 1
        #verinin ait oldugu cluster'i guncelliyoruz
        uye = liste_cluster_uye
    return liste_cluster_uye, centroid

sonuc, centroid = kernelKMeans(input, initMethod)
plot(sonuc, centroid, 1)
