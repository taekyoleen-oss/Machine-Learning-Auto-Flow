import React from 'react';
import { XMarkIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';

interface Sample {
  id?: number; // DB에서 로드할 때 사용
  filename: string;
  name: string;
  data: any;
  inputData?: string;
  description?: string;
  category?: string;
}

interface SamplesModalProps {
  isOpen: boolean;
  onClose: () => void;
  samples: Array<{ id?: number; filename: string; name: string; data: any; inputData?: string; description?: string; category?: string }>;
  onLoadSample: (sampleName: string, filename: string, sampleId?: number) => void;
  onManage?: () => void;
  isLoading?: boolean;
}

const CATEGORIES = ['전체', '머신러닝', '딥러닝', '통계분석', 'DFA', '프라이싱'] as const;

const SamplesModal: React.FC<SamplesModalProps> = ({
  isOpen,
  onClose,
  samples,
  onLoadSample,
  onManage,
  isLoading = false,
}) => {
  const [selectedCategory, setSelectedCategory] = React.useState<string>('전체');

  if (!isOpen) return null;

  const handleLoad = (sample: Sample) => {
    onLoadSample(sample.name, sample.filename, sample.id);
  };

  // 카테고리별 필터링
  const filteredSamples = selectedCategory === '전체'
    ? samples
    : samples.filter(sample => sample.category === selectedCategory);

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 rounded-lg shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="sticky top-0 bg-gray-900 border-b border-gray-700 z-10">
          <div className="p-4 flex justify-between items-center">
            <h2 className="text-2xl font-bold text-white">Samples</h2>
            <div className="flex items-center gap-2">
              {onManage && (
                <button
                  onClick={onManage}
                  className="text-gray-400 hover:text-white transition-colors p-2 rounded-md hover:bg-gray-800"
                  title="샘플 관리"
                >
                  <Cog6ToothIcon className="w-5 h-5" />
                </button>
              )}
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors p-1 rounded-md hover:bg-gray-800"
                aria-label="Close"
              >
                <XMarkIcon className="w-6 h-6" />
              </button>
            </div>
          </div>
          
          {/* 카테고리 필터 */}
          <div className="px-4 pb-4 flex gap-2 overflow-x-auto">
            {CATEGORIES.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-md text-sm font-semibold whitespace-nowrap transition-colors ${
                  selectedCategory === category
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>

        {/* 카드 그리드 */}
        <div className="flex-1 overflow-y-auto p-6">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-gray-400 text-lg">Loading samples...</div>
            </div>
          ) : filteredSamples.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-gray-400 text-lg">
                {selectedCategory === '전체' 
                  ? 'No samples available' 
                  : `'${selectedCategory}' 카테고리에 샘플이 없습니다.`}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredSamples.map((sample) => (
                <div
                  key={sample.filename}
                  className="bg-gray-800 rounded-lg border border-gray-700 p-6 hover:border-purple-500 transition-all duration-200 hover:shadow-lg hover:shadow-purple-500/20 flex flex-col"
                >
                  {/* 카드 헤더 */}
                  <h3 className="text-xl font-bold text-white mb-4 truncate">
                    {sample.name}
                  </h3>

                  {/* 모델 정보 */}
                  <div className="space-y-3 mb-4 flex-1">
                    <div>
                      <span className="text-gray-400 text-sm font-medium">
                        모델 파일:
                      </span>
                      <div className="text-white text-sm mt-1 truncate">
                        {sample.name}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400 text-sm font-medium">
                        입력데이터:
                      </span>
                      <div className="text-white text-sm mt-1">
                        {sample.inputData || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400 text-sm font-medium">
                        모델 설명:
                      </span>
                      <p className="text-gray-300 text-sm mt-1 line-clamp-3">
                        {sample.description || '설명 없음'}
                      </p>
                    </div>
                  </div>

                  {/* 실행 버튼 */}
                  <button
                    onClick={() => handleLoad(sample)}
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2.5 px-4 rounded-md transition-colors duration-200 mt-auto"
                  >
                    모델 실행
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SamplesModal;
