import React, { useState, useMemo } from 'react';
import { CodeBracketIcon, ClipboardIcon, CheckIcon } from './icons';
import { CanvasModule, Connection } from '../types';
import { generateFullPipelineCode } from '../utils/generatePipelineCode';
import { useTheme } from '../contexts/ThemeContext';

interface PipelineCodePanelProps {
    modules: CanvasModule[];
    connections: Connection[];
    isVisible: boolean;
    onToggle: () => void;
}

export const PipelineCodePanel: React.FC<PipelineCodePanelProps> = ({
    modules,
    connections,
    isVisible,
    onToggle
}) => {
    const { theme } = useTheme();
    const [copied, setCopied] = useState(false);

    const fullPipelineCode = useMemo(() => {
        if (modules.length === 0) {
            return '# 파이프라인이 비어있습니다. 모듈을 추가해주세요.';
        }
        return generateFullPipelineCode(modules, connections);
    }, [modules, connections]);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(fullPipelineCode);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy code:', err);
        }
    };

    return (
        <div 
            className={`absolute top-0 right-0 h-full bg-white dark:bg-gray-800 border-l border-gray-300 dark:border-gray-700 z-10 transition-transform duration-300 ease-in-out flex flex-col ${
                isVisible ? 'translate-x-0' : 'translate-x-full'
            }`}
            style={{ width: '400px' }}
        >
            {/* 헤더 */}
            <div className="flex items-center justify-between p-3 border-b border-gray-300 dark:border-gray-700 flex-shrink-0">
                <div className="flex items-center gap-2">
                    <CodeBracketIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                    <h3 className="text-sm font-bold text-gray-900 dark:text-white">전체 파이프라인 코드</h3>
                </div>
                <div className="flex items-center gap-2">
                    <button
                        onClick={handleCopy}
                        className="p-1.5 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
                        title="코드 복사"
                    >
                        {copied ? (
                            <CheckIcon className="w-4 h-4 text-green-600 dark:text-green-400" />
                        ) : (
                            <ClipboardIcon className="w-4 h-4 text-gray-700 dark:text-gray-300" />
                        )}
                    </button>
                </div>
            </div>

            {/* 코드 영역 */}
            <div className="flex-1 overflow-auto p-3">
                <pre className="bg-gray-100 dark:bg-gray-900 p-3 rounded-md overflow-x-auto text-xs font-mono text-gray-900 dark:text-gray-200 whitespace-pre-wrap">
                    <code>{fullPipelineCode}</code>
                </pre>
            </div>
        </div>
    );
};














































