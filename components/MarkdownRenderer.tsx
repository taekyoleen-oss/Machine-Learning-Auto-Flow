import React from 'react';

export const MarkdownRenderer: React.FC<{ text: string }> = ({ text }) => {
    const renderLine = (line: string) => {
        const boldRegex = /\*\*(.*?)\*\*/g;
        const codeRegex = /`([^`]+)`/g;
        const parts = [];
        let lastIndex = 0;
        let result;

        const combinedRegex = new RegExp(`(${boldRegex.source})|(${codeRegex.source})`, 'g');

        while ((result = combinedRegex.exec(line)) !== null) {
            // Text before the match
            if (result.index > lastIndex) {
                parts.push(line.substring(lastIndex, result.index));
            }
            // Matched part
            if (result[2]) { // Bold
                parts.push(<strong key={result.index}>{result[2]}</strong>);
            } else if (result[4]) { // Code
                parts.push(<code key={result.index} className="bg-gray-200 text-purple-700 px-1 py-0.5 rounded text-xs font-mono">{result[4]}</code>);
            }
            lastIndex = combinedRegex.lastIndex;
        }

        // Text after the last match
        if (lastIndex < line.length) {
            parts.push(line.substring(lastIndex));
        }

        return parts.length > 0 ? <>{parts}</> : <>{line}</>;
    };

    return (
        <div className="text-gray-700 space-y-3 font-sans text-base leading-relaxed">
            {text.split('\n').map((line, index) => {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('### ')) {
                    return <h4 key={index} className="text-lg font-semibold mt-4 mb-1 text-gray-800">{renderLine(trimmedLine.substring(4))}</h4>
                }
                if (trimmedLine.startsWith('## ')) {
                     return <h3 key={index} className="text-xl font-bold mt-5 mb-2 text-gray-900 border-b pb-1">{renderLine(trimmedLine.substring(3))}</h3>
                }
                 if (trimmedLine.startsWith('* ')) {
                    return <div key={index} className="flex items-start pl-4"><span className="mr-2 mt-1 text-purple-500">•</span><div className="flex-1">{renderLine(trimmedLine.substring(2))}</div></div>
                }
                if (trimmedLine === '') {
                    return null;
                }
                return <p key={index}>{renderLine(line)}</p>;
            })}
        </div>
    );
};
