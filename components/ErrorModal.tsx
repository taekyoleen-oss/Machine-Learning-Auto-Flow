import React, { useState, useEffect } from 'react';
import { XCircleIcon } from './icons';

interface ErrorModalProps {
  error: {
    moduleName: string;
    message: string;
    details?: string;
  } | null;
  onClose: () => void;
}

export const ErrorModal: React.FC<ErrorModalProps> = ({ error, onClose }) => {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (error) {
      setCopied(false);
    }
  }, [error]);

  if (!error) return null;

  const handleCopy = () => {
    const errorText = `Module: ${error.moduleName}\nError: ${error.message}${error.details ? `\nDetails:\n${error.details}` : ''}`;
    navigator.clipboard.writeText(errorText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div 
        className="bg-white text-gray-900 rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        <header className="flex items-center justify-between p-4 border-b border-gray-200 flex-shrink-0">
          <h2 className="text-xl font-bold text-red-600">Module Execution Error</h2>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-800">
            <XCircleIcon className="w-6 h-6" />
          </button>
        </header>
        <main className="flex-grow p-6 overflow-auto">
          <div className="space-y-4">
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Module:</h3>
              <p className="text-gray-900 font-mono text-sm">{error.moduleName}</p>
            </div>
            <div>
              <h3 className="text-sm font-semibold text-gray-700 mb-1">Error Message:</h3>
              <p className="text-gray-900 font-mono text-sm whitespace-pre-wrap break-words">{error.message}</p>
            </div>
            {error.details && (
              <div>
                <h3 className="text-sm font-semibold text-gray-700 mb-1">Details:</h3>
                <pre className="text-gray-900 font-mono text-xs whitespace-pre-wrap break-words bg-gray-50 p-3 rounded border border-gray-200 overflow-auto">
                  {error.details}
                </pre>
              </div>
            )}
          </div>
        </main>
        <footer className="flex items-center justify-end gap-2 p-4 border-t border-gray-200 flex-shrink-0">
          <button
            onClick={handleCopy}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 flex items-center gap-2"
          >
            {copied ? (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy Error
              </>
            )}
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            Close
          </button>
        </footer>
      </div>
    </div>
  );
};
