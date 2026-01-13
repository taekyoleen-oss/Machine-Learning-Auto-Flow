import React, { useState, useRef, useEffect, MouseEvent, TouchEvent } from 'react';
import { CanvasModule, ModuleType } from '../types';
import { XMarkIcon } from './icons';

interface ShapeRendererProps {
  module: CanvasModule;
  isSelected: boolean;
  onDragStart: (moduleId: string, e: MouseEvent | TouchEvent) => void;
  onDelete: (id: string) => void;
  onUpdateModule: (id: string, updates: Partial<CanvasModule>) => void;
  scale: number;
}

export const ShapeRenderer: React.FC<ShapeRendererProps> = ({
  module,
  isSelected,
  onDragStart,
  onDelete,
  onUpdateModule,
  scale,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [localText, setLocalText] = useState(module.shapeData?.text || '');
  const textInputRef = useRef<HTMLTextAreaElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  const [localName, setLocalName] = useState(module.name);

  useEffect(() => {
    setLocalText(module.shapeData?.text || '');
  }, [module.shapeData?.text]);

  useEffect(() => {
    setLocalName(module.name);
  }, [module.name]);

  useEffect(() => {
    if (isEditing && textInputRef.current) {
      textInputRef.current.focus();
    }
  }, [isEditing]);

  const handleDelete = (e: MouseEvent | TouchEvent) => {
    e.stopPropagation();
    onDelete(module.id);
  };

  const handleMouseDown = (e: MouseEvent) => {
    e.stopPropagation();
    onDragStart(module.id, e);
  };

  const handleTouchStart = (e: TouchEvent) => {
    e.stopPropagation();
    onDragStart(module.id, e);
  };

  const handleTextBlur = () => {
    setIsEditing(false);
    onUpdateModule(module.id, {
      shapeData: {
        ...module.shapeData,
        text: localText,
      },
    });
  };

  const handleNameBlur = () => {
    onUpdateModule(module.id, {
      name: localName,
    });
  };

  const handleNameKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      nameInputRef.current?.blur();
    }
  };

  const { position } = module;
  const componentStyle: React.CSSProperties = {
    transform: `translate(${position.x}px, ${position.y}px)`,
    userSelect: 'none',
  };

  const [isResizing, setIsResizing] = useState(false);
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, width: 0, height: 0 });
  const textBoxRef = useRef<HTMLDivElement>(null);
  const textBoxWidth = module.shapeData?.width || 200;
  const textBoxHeight = module.shapeData?.height || 100;
  const fontSize = module.shapeData?.fontSize || 14;

  const handleResizeStart = (e: MouseEvent | TouchEvent) => {
    e.stopPropagation();
    setIsResizing(true);
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
    const clientY = 'touches' in e ? e.touches[0].clientY : e.clientY;
    setResizeStart({
      x: clientX,
      y: clientY,
      width: textBoxWidth,
      height: textBoxHeight,
    });
  };

  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: globalThis.MouseEvent) => {
      const dx = e.clientX - resizeStart.x;
      const dy = e.clientY - resizeStart.y;
      const newWidth = Math.max(150, resizeStart.width + dx);
      const newHeight = Math.max(80, resizeStart.height + dy);
      
      onUpdateModule(module.id, {
        shapeData: {
          ...module.shapeData,
          width: newWidth,
          height: newHeight,
        },
      });
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing, resizeStart, module.id, module.shapeData, onUpdateModule]);

  if (module.type === ModuleType.TextBox) {
    return (
      <div
        ref={textBoxRef}
        style={{
          ...componentStyle,
          width: `${textBoxWidth}px`,
          height: `${textBoxHeight}px`,
        }}
        className={`absolute bg-green-900/30 border-2 border-green-500 rounded-lg shadow-lg ${
          isSelected ? 'ring-2 ring-offset-2 ring-offset-gray-900 ring-green-500' : ''
        } cursor-move`}
        onMouseDown={handleMouseDown}
        onTouchStart={handleTouchStart}
      >
        {isSelected && (
          <>
            <button
              onClick={handleDelete}
              onTouchEnd={handleDelete}
              className="absolute -top-2 -right-2 z-20 p-0.5 bg-gray-700 rounded-full text-gray-300 hover:bg-red-600 hover:text-white transition-colors"
              title="Delete"
            >
              <XMarkIcon className="w-3.5 h-3.5" />
            </button>
            {/* Resize handle */}
            <div
              onMouseDown={handleResizeStart}
              onTouchStart={handleResizeStart}
              className="absolute bottom-0 right-0 w-4 h-4 cursor-nwse-resize bg-gray-600 hover:bg-gray-500 rounded-tl-lg"
              style={{
                clipPath: 'polygon(100% 0, 0 0, 100% 100%)',
              }}
            />
          </>
        )}
        <div className="p-2 w-full h-full flex flex-col">
          {isEditing ? (
            <textarea
              ref={textInputRef}
              value={localText}
              onChange={(e) => setLocalText(e.target.value)}
              onBlur={handleTextBlur}
              onClick={(e) => e.stopPropagation()}
              className="flex-1 w-full px-2 py-1 bg-gray-900 text-white rounded border border-green-600 focus:border-green-500 focus:outline-none resize-none whitespace-pre-wrap"
              placeholder="텍스트를 입력하세요..."
              style={{ wordWrap: 'break-word', whiteSpace: 'pre-wrap', fontSize: `${fontSize}px` }}
            />
          ) : (
            <div
              onClick={(e) => {
                e.stopPropagation();
                setIsEditing(true);
              }}
              className="flex-1 w-full px-2 py-1 bg-gray-900 text-white rounded border border-transparent hover:border-green-600 cursor-text overflow-auto whitespace-pre-wrap"
              style={{ wordWrap: 'break-word', whiteSpace: 'pre-wrap', fontSize: `${fontSize}px` }}
            >
              {localText || <span className="text-gray-500">텍스트를 입력하세요...</span>}
            </div>
          )}
        </div>
      </div>
    );
  }

  if (module.type === ModuleType.GroupBox) {
    const bounds = module.shapeData?.bounds;
    if (!bounds) {
      return null; // Bounds not calculated yet
    }

    // Group box position is already adjusted in App.tsx
    // Name is positioned above the box
    const nameY = bounds.y - 25;

    return (
      <>
        {/* Group name outside the box */}
        <div
          style={{
            transform: `translate(${bounds.x}px, ${nameY}px)`,
            userSelect: 'none',
          }}
          className="absolute"
        >
          <input
            ref={nameInputRef}
            type="text"
            value={localName}
            onChange={(e) => setLocalName(e.target.value)}
            onBlur={handleNameBlur}
            onKeyDown={handleNameKeyDown}
            onClick={(e) => e.stopPropagation()}
            className="px-2 py-1 bg-gray-800/90 text-white text-sm rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
            placeholder="그룹 이름"
            style={{ minWidth: '120px' }}
          />
        </div>
        {/* Group box */}
        <div
          style={{
            width: `${bounds.width}px`,
            height: `${bounds.height}px`,
            transform: `translate(${bounds.x}px, ${bounds.y}px)`,
            userSelect: 'none',
          }}
          className={`absolute border-2 border-purple-500 bg-purple-500/10 rounded-lg ${
            isSelected ? 'ring-2 ring-offset-2 ring-offset-gray-900 ring-purple-500' : ''
          } cursor-move`}
          onMouseDown={handleMouseDown}
          onTouchStart={handleTouchStart}
        >
          {isSelected && (
            <button
              onClick={handleDelete}
              onTouchEnd={handleDelete}
              className="absolute -top-2 -right-2 z-20 p-0.5 bg-gray-700 rounded-full text-gray-300 hover:bg-red-600 hover:text-white transition-colors"
              title="Delete Group"
            >
              <XMarkIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </>
    );
  }

  return null;
};

