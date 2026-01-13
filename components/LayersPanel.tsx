import React from 'react';
import { CanvasModule } from '../types';
import { TOOLBOX_MODULES } from '../constants';
import { Bars3Icon } from './icons';

interface LayersPanelProps {
    components: CanvasModule[];
    onSelect: (id: string) => void;
    selectedComponentIds: string[]; // Keep for highlighting
}

const LayerItem: React.FC<{
    component: CanvasModule;
    level: number;
    isSelected: boolean;
    onSelect: (id: string) => void;
}> = ({ component, level, isSelected, onSelect }) => {
    
    const { icon: Icon } = TOOLBOX_MODULES.find(m => m.type === component.type) || {};

    return (
        <div
            onClick={() => onSelect(component.id)}
            className={`flex items-center px-4 py-2 text-sm cursor-pointer truncate ${isSelected ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-800'}`}
            style={{ paddingLeft: `${1 + level * 1.5}rem` }}
        >
            {Icon && <Icon className="w-4 h-4 mr-2 flex-shrink-0" />}
            <span className="truncate">{component.name}</span>
        </div>
    );
};


export const LayersPanel: React.FC<LayersPanelProps> = ({ components, onSelect, selectedComponentIds }) => {
    return (
        <aside className="w-64 bg-gray-900 border-r border-gray-700 flex-shrink-0 flex flex-col">
            <div className="p-3 border-b border-gray-700 flex-shrink-0 flex items-center gap-2">
                 <Bars3Icon className="w-5 h-5" />
                 <h3 className="text-lg font-semibold text-white">Outline</h3>
            </div>
            <div className="flex-grow overflow-y-auto panel-scrollbar">
                {components.length === 0 ? (
                    <div className="px-4 py-2 text-sm text-gray-500">No modules on canvas.</div>
                ) : (
                    [...components].map((component) => (
                        <LayerItem
                            key={component.id}
                            component={component}
                            level={0}
                            isSelected={selectedComponentIds.includes(component.id)}
                            onSelect={onSelect}
                        />
                    ))
                )}
            </div>
        </aside>
    );
};
