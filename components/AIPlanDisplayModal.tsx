import React, { useState, useEffect } from 'react';
import { XCircleIcon, SparklesIcon } from './icons';
import { MarkdownRenderer } from './MarkdownRenderer';

interface AIPlanDisplayModalProps {
    isOpen: boolean;
    onClose: () => void;
    plan: string;
    onCreatePipeline?: () => void;
    hasPipelineData?: boolean;
    isLoading?: boolean;
}

export const AIPlanDisplayModal: React.FC<AIPlanDisplayModalProps> = ({ 
    isOpen, 
    onClose, 
    plan, 
    onCreatePipeline,
    hasPipelineData = false,
    isLoading = false
}) => {
    const [displayedPlan, setDisplayedPlan] = useState('');
    const [isTyping, setIsTyping] = useState(false);

    useEffect(() => {
        if (plan && !isLoading) {
            // 결과가 도착하면 순차적으로 표시
            setIsTyping(true);
            setDisplayedPlan('');
            let currentIndex = 0;
            
            const typeInterval = setInterval(() => {
                if (currentIndex < plan.length) {
                    // 한 번에 여러 문자를 표시하여 더 빠르게
                    const chunkSize = Math.max(1, Math.floor(plan.length / 100));
                    const nextIndex = Math.min(currentIndex + chunkSize, plan.length);
                    setDisplayedPlan(plan.substring(0, nextIndex));
                    currentIndex = nextIndex;
                } else {
                    setIsTyping(false);
                    clearInterval(typeInterval);
                }
            }, 20); // 20ms마다 업데이트

            return () => clearInterval(typeInterval);
        } else if (!plan && isLoading) {
            setDisplayedPlan('');
            setIsTyping(false);
        }
    }, [plan, isLoading]);

    if (!isOpen) return null;

    return (
        <div 
            className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50"
            onClick={isLoading ? undefined : onClose}
        >
            <div 
                className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] flex flex-col"
                onClick={e => e.stopPropagation()}
            >
                <header className="flex items-center justify-between p-4 border-b border-gray-200">
                    <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                        <SparklesIcon className="w-6 h-6 text-purple-500" />
                        {isLoading ? '데이터 분석 중...' : '데이터 분석 결과'}
                    </h2>
                    {!isLoading && (
                        <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
                            <XCircleIcon className="w-6 h-6" />
                        </button>
                    )}
                </header>
                <main className="p-6 overflow-y-auto flex-1">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center py-12">
                            <div className="relative w-16 h-16 mb-4">
                                <div className="absolute inset-0 border-4 border-indigo-200 rounded-full"></div>
                                <div className="absolute inset-0 border-4 border-transparent border-t-indigo-600 rounded-full animate-spin"></div>
                            </div>
                            <p className="text-gray-600 text-lg">데이터를 분석하고 있습니다...</p>
                            <p className="text-gray-500 text-sm mt-2">잠시만 기다려주세요</p>
                        </div>
                    ) : displayedPlan ? (
                        <div>
                            <MarkdownRenderer text={displayedPlan} />
                            {isTyping && (
                                <span className="inline-block w-2 h-5 bg-indigo-600 ml-1 animate-pulse"></span>
                            )}
                        </div>
                    ) : (
                        <div className="text-gray-500 text-center py-8">
                            분석 결과가 없습니다.
                        </div>
                    )}
                </main>
                <footer className="flex justify-end gap-2 p-4 bg-gray-50 border-t rounded-b-lg">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-semibold text-gray-700 bg-gray-200 hover:bg-gray-300 rounded-md"
                    >
                        닫기
                    </button>
                    {hasPipelineData && onCreatePipeline && (
                        <button
                            onClick={onCreatePipeline}
                            className="px-4 py-2 text-sm font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-md"
                        >
                            파이프라인 생성하기
                        </button>
                    )}
                </footer>
            </div>
        </div>
    );
};
