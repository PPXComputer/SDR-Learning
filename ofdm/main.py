import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy

from enum import Enum


class ModulationType(Enum):
    BPSK = 1
    QPSK = 2
    PSK8 = 3
    QAM16 = 4
    QAM64 = 5

    @staticmethod
    def modulation(bits: np.ndarray, modulation_type: Enum):
        # 调制解调的方式 将更多位的信息存放于 信息上
        # 其中的 位数 就是 在 +1 -1 上进行叠加
        if modulation_type == ModulationType.QPSK:
            modem = cpy.PSKModem(4)
        elif modulation_type == ModulationType.QAM64:
            modem = cpy.QAMModem(64)
        elif modulation_type == ModulationType.QAM16:
            modem = cpy.QAMModem(16)
        elif modulation_type == ModulationType.PSK8:
            modem = cpy.PSKModem(8)

        elif modulation_type == ModulationType.BPSK:
            modem = cpy.PSKModem(2)
        else:
            raise RuntimeError('')
        return modem.modulate(bits), modem


class ChannelType(Enum):
    random = 1
    awgn = 2


class Property:
    subcarrierNumber = 64  # 子载波数量
    cyclicPrefixLength = subcarrierNumber // 4  # 循环前缀长度
    pLength = 8  # 导频数
    pilotValue = 3 + 3j  # 导频格式
    modulationType = ModulationType.QAM16  # 调制方式
    channelType = ChannelType.random  # 噪声形式
    noiseDB = 25  # 接收端的 信噪比
    allCarriers = np.arange(subcarrierNumber)  # 子载波编码
    pilotCarrier = allCarriers[::pLength]  # 每间隔 P个子载波一个导频

    # 为了估计当前的信道 ,将最后一个子载波也作为导频 pilotCarrierArray = [pilotCarrier.flatten() ,allCarriers[-1]]
    pilotCarrierArray = np.hstack([pilotCarrier, np.array(allCarriers[-1])])
    realPLength = pLength + 1  # 真实的导频数也加 一

    pass


def showPlotGraph():
    carries = np.delete(Property.allCarriers, Property.pilotCarrier)
    plt.figure(figsize=(40, 4))
    plt.plot(Property.pilotCarrier, np.zeros_like(Property.pilotCarrier), 'bo', label='pilot')
    plt.plot(carries, np.zeros_like(carries), 'ro', label='data')
    plt.legend(fontsize=10, ncol=2)
    plt.xlim((-1, Property.subcarrierNumber))
    plt.ylim((-0.1, 0.3))
    plt.xlabel('Carrier Index')
    plt.yticks([])
    plt.grid(True)
    plt.savefig('carrier.png')


def visualTheChannel():
    channelResponse = np.array([1, 0, 0.3 + 0.3j])
    # 将子载波长度作为的当前傅里叶变化的长度
    hExact = np.fft.fft(channelResponse, Property.subcarrierNumber)
    plt.plot(Property.allCarriers, abs(hExact))  # 获得傅里叶变化结果的虚部
    plt.xlabel('SubCarrier Index')
    plt.ylabel('H(f)')
    plt.grid(True)
    plt.xlim(0, Property.subcarrierNumber - 1)
    plt.show()
    # t = np.arange(256)
    # sp = np.fft.fft(np.sin(t))
    # freq = np.fft.fftfreq(t.shape[-1])
    # plt.plot(freq, sp.real, freq, sp.imag)
    # plt.show()
    pass


# x_s 为输入的复数形式的 功率
# snrDb 信噪比
def transferInChannel(x_s, snrDB):
    data_pwr = np.mean(abs(x_s ** 2))  # 数据功率
    noise_pwr = data_pwr / (10 ** (snrDB / 10))  # 噪声功率 数据功率 / 噪声比
    # 噪声   1/根号(2)  *  (噪声A+噪声B*j) *(频幅)
    noise = 1 / np.sqrt(2) * \
            (np.random.rand(len(x_s)) + 1j *
             np.random.rand(len(x_s))) * np.sqrt(noise_pwr)
    return x_s + noise, noise_pwr


def channel(in_signal, snrDB, channelType: Enum):
    chanenelResponse = np.array([1, 0.3 + 0.3j])
    if channelType == ChannelType.random:
        # 卷积运算
        convolved = np.convolve(in_signal, chanenelResponse)
        out_signal, noise_pwr = transferInChannel(convolved, snrDB)
    else:
        out_signal, noise_pwr = transferInChannel(in_signal, snrDB)
    return out_signal, noise_pwr


# 每个 ofdm  符合的有效载荷位数
def eachOfdmInPayloadDigits(data_carriers, modulation_type: ModulationType):
    if modulation_type == ModulationType.QAM16:
        mu = 4
    elif modulation_type == ModulationType.QPSK:
        mu = 2
    elif modulation_type == ModulationType.BPSK:
        mu = 1
    elif modulation_type == ModulationType.QAM64:
        mu = 6
    elif modulation_type == ModulationType.PSK8:
        mu = 3
    else:
        raise RuntimeError('')
    return mu * len(data_carriers)


def processData(carries):
    bits = np.random.binomial(n=1, p=0.5, size=eachOfdmInPayloadDigits(carries, ModulationType.QAM16))

    qam_s, modem = ModulationType.modulation(bits, Property.modulationType)

    def OFDM_symbol(qam_payload):
        # 子载波位置
        # 在导频位置上插入 导频 最后一个载波作为导频使用了 所以载波数减1
        symbol = np.zeros(Property.subcarrierNumber, dtype=complex)
        symbol[Property.pilotCarrierArray] = Property.pilotValue  # 在导频段填上 导频格式\
        # 在数据位置上插入 数据
        dataCarries = np.delete(Property.allCarriers, Property.pilotCarrierArray)
        symbol[carries] = qam_payload
        return symbol

    ofdm_data = OFDM_symbol(qam_s)
    # 逆离散傅里叶变换
    ofdm_time = np.fft.ifft(ofdm_data)
    # 添加循环前缀
    # 将 后几项元素移动
    prefix = ofdm_time[-Property.cyclicPrefixLength:]
    ofdm_time_with_cp = np.hstack([prefix, ofdm_time])
    return ofdm_time_with_cp, qam_s


#  通过信道
def throughoutChannel(ofdmWithCp):
    ofdm_rx, noise_pwr = channel(ofdmWithCp, Property.noiseDB, ChannelType.random)
    ofdm_tx = ofdmWithCp
    plt.figure(figsize=(14, 4))
    plt.plot(abs(ofdm_tx), label='TX Signal')
    plt.plot(abs(ofdm_rx), label='RX Signal')
    plt.legend(fontsize=10)
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.grid(True)
    plt.savefig('throughoutChannel.png')

    return ofdm_rx


def sendProcess():
    carries = np.delete(Property.allCarriers, Property.pilotCarrierArray)
    ofdmWithCp, qam_s = processData(carries)
    ofdm_rx = throughoutChannel(ofdmWithCp)
    return ofdm_rx, ofdmWithCp, qam_s


# 信道估计
def channelEstimate(ofdm_demod, channelResponse):
    # 取导频的数据
    pilots = ofdm_demod[Property.pilotCarrierArray]
    # LS信道估计
    hest_at_pilots = pilots / Property.pilotValue

    hest_abs = interpolate.interp1d(Property.pilotCarrierArray, abs(hest_at_pilots), kind='linear')(
        Property.allCarriers)
    hest_phase = interpolate.interp1d(Property.pilotCarrierArray, np.angle(hest_at_pilots), kind='linear')(
        Property.allCarriers)
    hest = hest_abs * np.exp(1j * hest_phase)
    hExact = np.fft.fft(channelResponse, Property.subcarrierNumber)
    plt.plot(Property.allCarriers, abs(hExact), label='Correct Channel')
    plt.scatter(Property.pilotCarrierArray, abs(hest_at_pilots), label='pilot estimates')
    plt.plot(Property.allCarriers, abs(hest), label=u'通过插值估计渠道')
    plt.grid(True)
    plt.xlabel('Carrier index ')
    plt.ylabel('H(f)')
    plt.legend(fontsize=10)
    plt.ylim(0.2)
    plt.savefig('信道估计.png')
    return hest


def reviveProcess():
    ofdm_rx, ofdm_tx, qam_s = sendProcess()
    cp = Property.cyclicPrefixLength

    # 去除前缀后进行快速傅里叶变化
    ofdmRxWithOutCp = ofdm_rx[cp:(cp + Property.subcarrierNumber)]
    ofdm_demod = np.fft.fft(ofdmRxWithOutCp)
    # 信道估计
    estimate = channelEstimate(ofdm_demod, channelResponse=np.array([1, 0, 0.3 + 0.3j]))
    #  信道均衡
    equalized_hest = equalize(ofdm_demod, estimate)
    carries = np.delete(Property.allCarriers, Property.pilotCarrier)
    # 获取数据位置信息
    QAM_est = equalized_hest[carries]
    plt.cla()
    plt.plot(QAM_est.real, QAM_est.imag, 'bo')
    #  比特信号调制
    plt.plot(qam_s.real, qam_s.imag, 'ro')
    plt.grid(True)
    plt.xlabel('实部')
    plt.ylabel('虚部')
    plt.title("接收的星云图")
    plt.savefig('map.png')


#  信道均衡
def equalize(ofdm_demod, estimate):
    return ofdm_demod / estimate


if __name__ == '__main__':
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # showPlotGraph()
    # visualTheChannel()
    reviveProcess()
    pass
