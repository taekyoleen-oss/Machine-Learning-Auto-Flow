import React, { useState } from 'react';
import { CanvasModule, NormalityCheckerOutput } from '../types';
import { XCircleIcon } from './icons';

interface NormalityCheckerPreviewModalProps {
    module: CanvasModule;
    projectName: string;
    onClose: () => void;
}

export const NormalityCheckerPreviewModal: React.FC<NormalityCheckerPreviewModalProps> = ({ 
    module, 
    projectName, 
    onClose
}) => {
    const output = module.outputData as NormalityCheckerOutput;
    if (!output || output.type !== 'NormalityCheckerOutput') return null;

    const [activeTab, setActiveTab] = useState<'summary' | 'qqplot' | 'ecdf' | 'boxplot'>('summary');

    const { column, skewness, kurtosis, jarqueBera, testResults, histogramImage, qqPlotImage, ecdfImage, boxplotImage } = output;

    const formatValue = (value: any): string => {
        if (value === null || value === undefined) return 'N/A';
        if (typeof value === 'number') {
            if (Math.abs(value) < 0.0001 || Math.abs(value) > 1000000) {
                return value.toExponential(4);
            }
            return value.toFixed(4);
        }
        return String(value);
    };

    const getConclusionColor = (conclusion?: string): string => {
        if (!conclusion) return 'text-gray-400';
        if (conclusion.includes('Reject')) return 'text-red-400';
        if (conclusion.includes('Fail to reject')) return 'text-green-400';
        return 'text-gray-400';
    };

    // 각 통계량에 대한 설명
    const getStatisticDescription = (measure: string): string => {
        switch (measure) {
            case 'Skewness':
                return '데이터의 비대칭 정도를 측정합니다. 0에 가까울수록 대칭적입니다.';
            case 'Kurtosis':
                return '데이터의 꼬리 두께를 측정합니다. 정규분포는 0에 가까운 값을 가집니다.';
            case 'Jarque-Bera Statistic':
                return '왜도와 첨도를 기반으로 한 정규성 검정 통계량입니다.';
            default:
                return '';
        }
    };

    // 각 검정에 대한 설명
    const getTestDescription = (testType: string): string => {
        switch (testType) {
            case 'shapiro_wilk':
                return '가장 널리 사용되는 정규성 검정입니다. 소표본에 강합니다.';
            case 'kolmogorov_smirnov':
                return '분포 전체를 비교하는 검정입니다. 큰 표본에 적합합니다.';
            case 'anderson_darling':
                return '꼬리 부분에 민감한 검정입니다. Tail 분석에 강합니다.';
            case 'dagostino_k2':
                return '왜도·첨도 기반 검정입니다. 빠르고 직관적입니다.';
            case 'jarque_bera':
                return '왜도와 첨도를 기반으로 한 정규성 검정입니다.';
            default:
                return '';
        }
    };

    const tabs = [
        { id: 'summary' as const, label: 'Summary' },
        { id: 'qqplot' as const, label: 'Q-Q Plot' },
        { id: 'ecdf' as const, label: 'ECDF vs Normal CDF' },
        { id: 'boxplot' as const, label: 'Boxplot' },
    ];

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
            onClick={onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-6xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
                    <h2 className="text-xl font-bold text-gray-800">Normality Checker: {module.name}</h2>
                    <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                        <XCircleIcon className="w-6 h-6" />
                    </button>
                </header>
                
                <div className="flex-shrink-0 border-b border-gray-200">
                    <div className="flex">
                        {tabs.map(tab => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex-1 px-4 py-3 text-sm font-semibold transition-colors ${
                                    activeTab === tab.id
                                        ? 'bg-gray-700 text-white border-b-2 border-blue-500'
                                        : 'text-gray-400 hover:bg-gray-700/50'
                                }`}
                            >
                                {tab.label}
                            </button>
                        ))}
                    </div>
                </div>
                
                <main className="flex-grow p-6 overflow-auto">
                    {activeTab === 'summary' && (
                        <div className="space-y-6">
                            {/* Column name */}
                            <div className="text-lg font-semibold text-gray-800">
                                Column: <span className="font-mono text-blue-600">{column}</span>
                            </div>

                            {/* Statistics Table and Test Results */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                                {/* Left: Statistics Table */}
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <h3 className="text-md font-bold text-gray-800 mb-4">Statistics</h3>
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="border-b border-gray-300">
                                                <th className="text-left py-2 px-3 font-semibold">Measure</th>
                                                <th className="text-right py-2 px-3 font-semibold">Value</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr className="border-b border-gray-200">
                                                <td className="py-2 px-3">
                                                    <div>Skewness</div>
                                                    <div className="text-xs text-gray-500 mt-1">{getStatisticDescription('Skewness')}</div>
                                                </td>
                                                <td className="text-right py-2 px-3 font-mono">{formatValue(skewness)}</td>
                                            </tr>
                                            <tr className="border-b border-gray-200">
                                                <td className="py-2 px-3">
                                                    <div>Kurtosis</div>
                                                    <div className="text-xs text-gray-500 mt-1">{getStatisticDescription('Kurtosis')}</div>
                                                </td>
                                                <td className="text-right py-2 px-3 font-mono">{formatValue(kurtosis)}</td>
                                            </tr>
                                            <tr>
                                                <td className="py-2 px-3">
                                                    <div>Jarque-Bera Statistic</div>
                                                    <div className="text-xs text-gray-500 mt-1">{getStatisticDescription('Jarque-Bera Statistic')}</div>
                                                </td>
                                                <td className="text-right py-2 px-3 font-mono">{formatValue(jarqueBera.statistic)}</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>

                                {/* Right: Test Results */}
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <h3 className="text-md font-bold text-gray-800 mb-4">Test Results</h3>
                                    <div className="space-y-3">
                                        {/* Jarque-Bera Test */}
                                        <div className="bg-white rounded p-3 border border-gray-200">
                                            <div className="text-sm font-semibold text-gray-700 mb-1">Jarque-Bera Test</div>
                                            <div className="text-xs text-gray-500 mb-2">{getTestDescription('jarque_bera')}</div>
                                            <div className="grid grid-cols-2 gap-2 text-xs">
                                                <div>
                                                    <span className="text-gray-500">p-value:</span>
                                                    <span className={`ml-2 font-mono ${
                                                        jarqueBera.pValue < 0.05 ? 'text-red-600' : 'text-green-600'
                                                    }`}>
                                                        {formatValue(jarqueBera.pValue)}
                                                    </span>
                                                </div>
                                                <div className={getConclusionColor(jarqueBera.conclusion)}>
                                                    {jarqueBera.conclusion}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Other Tests */}
                                        {testResults.map((test, index) => (
                                            <div key={index} className="bg-white rounded p-3 border border-gray-200">
                                                <div className="text-sm font-semibold text-gray-700 mb-1">{test.testName}</div>
                                                <div className="text-xs text-gray-500 mb-2">{getTestDescription(test.testType)}</div>
                                                <div className="grid grid-cols-2 gap-2 text-xs">
                                                    {test.statistic !== undefined && (
                                                        <div>
                                                            <span className="text-gray-500">Statistic:</span>
                                                            <span className="ml-2 font-mono">{formatValue(test.statistic)}</span>
                                                        </div>
                                                    )}
                                                    {test.pValue !== undefined ? (
                                                        <div>
                                                            <span className="text-gray-500">p-value:</span>
                                                            <span className={`ml-2 font-mono ${
                                                                test.pValue < 0.05 ? 'text-red-600' : 'text-green-600'
                                                            }`}>
                                                                {formatValue(test.pValue)}
                                                            </span>
                                                        </div>
                                                    ) : test.criticalValue !== undefined && (
                                                        <div>
                                                            <span className="text-gray-500">Critical Value:</span>
                                                            <span className="ml-2 font-mono">{formatValue(test.criticalValue)}</span>
                                                        </div>
                                                    )}
                                                    {test.conclusion && (
                                                        <div className={`col-span-2 ${getConclusionColor(test.conclusion)}`}>
                                                            {test.conclusion}
                                                        </div>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>

                            {/* Histogram with Normal Curve Overlay */}
                            {histogramImage && (
                                <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                                    <h3 className="text-md font-bold text-gray-800 mb-4">Histogram + Normal Curve Overlay</h3>
                                    <div className="flex justify-center">
                                        <img 
                                            src={`data:image/png;base64,${histogramImage}`} 
                                            alt="Histogram with Normal Curve" 
                                            className="max-w-full h-auto rounded"
                                        />
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {activeTab === 'qqplot' && qqPlotImage && (
                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <h3 className="text-md font-bold text-gray-800 mb-4">Q-Q Plot</h3>
                            <div className="flex justify-center">
                                <img 
                                    src={`data:image/png;base64,${qqPlotImage}`} 
                                    alt="Q-Q Plot" 
                                    className="max-w-full h-auto rounded"
                                />
                            </div>
                        </div>
                    )}

                    {activeTab === 'ecdf' && ecdfImage && (
                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <h3 className="text-md font-bold text-gray-800 mb-4">ECDF vs Normal CDF</h3>
                            <div className="flex justify-center">
                                <img 
                                    src={`data:image/png;base64,${ecdfImage}`} 
                                    alt="ECDF vs Normal CDF" 
                                    className="max-w-full h-auto rounded"
                                />
                            </div>
                        </div>
                    )}

                    {activeTab === 'boxplot' && boxplotImage && (
                        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
                            <h3 className="text-md font-bold text-gray-800 mb-4">Boxplot</h3>
                            <div className="flex justify-center">
                                <img 
                                    src={`data:image/png;base64,${boxplotImage}`} 
                                    alt="Boxplot" 
                                    className="max-w-full h-auto rounded"
                                />
                            </div>
                        </div>
                    )}
                </main>
            </div>
        </div>
    );
};
