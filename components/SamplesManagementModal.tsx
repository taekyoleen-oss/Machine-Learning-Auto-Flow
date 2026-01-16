import React, { useState, useEffect } from "react";
import {
  XMarkIcon,
  PencilIcon,
  TrashIcon,
  ArrowUpTrayIcon,
} from "@heroicons/react/24/outline";
import { samplesApi, Sample } from "../utils/samples-api";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onRefresh: () => void;
}

export const SamplesManagementModal: React.FC<Props> = ({
  isOpen,
  onClose,
  onRefresh,
}) => {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [editing, setEditing] = useState<Sample | null>(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    input_data: "",
    description: "",
    category: "머신러닝",
  });

  useEffect(() => {
    if (isOpen) {
      loadSamples();
    }
  }, [isOpen]);

  const loadSamples = async () => {
    setLoading(true);
    try {
      const data = await samplesApi.getAll();
      setSamples(data);
    } catch (error: any) {
      console.error("Failed to load samples:", error);
      // 404 에러인 경우 서버가 실행되지 않았을 가능성
      if (error.message && error.message.includes("404")) {
        alert(
          "샘플 관리 서버에 연결할 수 없습니다.\n\n" +
            "서버를 시작하려면 다음 명령어를 실행하세요:\n" +
            "npm run server\n\n" +
            "또는 터미널에서:\n" +
            "node server/split-data-server.js"
        );
      } else {
        alert("샘플 목록을 불러오는데 실패했습니다: " + error.message);
      }
      setSamples([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("정말 이 샘플을 삭제하시겠습니까?")) return;

    try {
      await samplesApi.delete(id);
      await loadSamples();
      onRefresh();
      alert("삭제되었습니다.");
    } catch (error: any) {
      alert("삭제 실패: " + error.message);
    }
  };

  const handleEdit = (sample: Sample) => {
    setEditing(sample);
    setFormData({
      name: sample.name,
      input_data: sample.input_data || "",
      description: sample.description || "",
      category: sample.category || "머신러닝",
    });
  };

  const handleSave = async () => {
    if (!editing || !editing.id) return;

    // 필수 필드 검증
    if (!formData.name || formData.name.trim() === "") {
      alert("이름을 입력해주세요.");
      return;
    }

    try {
      setLoading(true);
      await samplesApi.update(editing.id, formData);
      setEditing(null);
      setFormData({
        name: "",
        input_data: "",
        description: "",
        category: "머신러닝",
      });
      await loadSamples();
      onRefresh();
      alert("저장되었습니다.");
    } catch (error: any) {
      alert("저장 실패: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    if (
      editing &&
      (formData.name !== (editing.name || "") ||
        formData.input_data !== (editing.input_data || "") ||
        formData.description !== (editing.description || "") ||
        formData.category !== (editing.category || "머신러닝"))
    ) {
      if (
        !confirm(
          "변경사항이 있습니다. 정말 취소하시겠습니까? 변경사항은 저장되지 않습니다."
        )
      ) {
        return;
      }
    }
    setEditing(null);
    setFormData({
      name: "",
      input_data: "",
      description: "",
      category: "머신러닝",
    });
  };

  const handleFileImport = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.endsWith(".ins") && !file.name.endsWith(".json")) {
      alert("지원하는 파일 형식은 .ins 또는 .json입니다.");
      return;
    }

    try {
      setLoading(true);
      await samplesApi.importFromFile(file);
      await loadSamples();
      onRefresh();
      alert("파일이 성공적으로 가져와졌습니다.");
      // 파일 입력 초기화
      e.target.value = "";
    } catch (error: any) {
      alert("가져오기 실패: " + error.message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-70 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 rounded-lg shadow-2xl w-full max-w-5xl max-h-[90vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="p-4 border-b border-gray-700 flex justify-between items-center flex-shrink-0">
          <h2 className="text-2xl font-bold text-white">샘플 관리</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors p-1 rounded-md hover:bg-gray-800"
            aria-label="Close"
          >
            <XMarkIcon className="w-6 h-6" />
          </button>
        </div>

        {/* 파일 가져오기 */}
        <div className="p-4 border-b border-gray-700 flex-shrink-0">
          <label className="block mb-2 text-sm font-medium text-gray-300">
            파일에서 가져오기 (.ins, .json)
          </label>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-md cursor-pointer transition-colors">
              <ArrowUpTrayIcon className="w-5 h-5" />
              <span>파일 선택</span>
              <input
                type="file"
                accept=".ins,.json"
                onChange={handleFileImport}
                className="hidden"
                disabled={loading}
              />
            </label>
            {loading && (
              <span className="text-gray-400 text-sm">처리 중...</span>
            )}
          </div>
        </div>

        {/* 샘플 목록 */}
        <div className={`flex-1 overflow-y-auto p-4 ${editing ? "pb-0" : ""}`}>
          {loading && samples.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-gray-400 text-lg">로딩 중...</div>
            </div>
          ) : samples.length === 0 ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-gray-400 text-lg">샘플이 없습니다</div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700">
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      이름
                    </th>
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      카테고리
                    </th>
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      입력 데이터
                    </th>
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      설명
                    </th>
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      생성일
                    </th>
                    <th className="text-left p-3 text-gray-300 font-semibold">
                      작업
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((sample) => (
                    <tr
                      key={sample.id}
                      className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors"
                    >
                      <td className="p-3 text-white font-medium">
                        {sample.name}
                      </td>
                      <td className="p-3 text-gray-400">
                        <span className="px-2 py-1 bg-purple-600/20 text-purple-300 rounded text-xs">
                          {sample.category || "머신러닝"}
                        </span>
                      </td>
                      <td className="p-3 text-gray-400">
                        {sample.input_data || "-"}
                      </td>
                      <td className="p-3 text-gray-400 max-w-md truncate">
                        {sample.description || "-"}
                      </td>
                      <td className="p-3 text-gray-500 text-xs">
                        {sample.created_at
                          ? new Date(sample.created_at).toLocaleDateString(
                              "ko-KR"
                            )
                          : "-"}
                      </td>
                      <td className="p-3">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => handleEdit(sample)}
                            className="text-blue-400 hover:text-blue-300 transition-colors p-1 rounded"
                            title="수정"
                          >
                            <PencilIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(sample.id!)}
                            className="text-red-400 hover:text-red-300 transition-colors p-1 rounded"
                            title="삭제"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* 수정 폼 - 하단 고정 */}
        {editing && (
          <div className="p-4 border-t-2 border-purple-600 bg-gray-800 flex-shrink-0 shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">샘플 수정</h3>
              <button
                onClick={handleCancel}
                className="text-gray-400 hover:text-white transition-colors"
                title="닫기"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-300 mb-1">이름</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">
                  입력 데이터
                </label>
                <input
                  type="text"
                  value={formData.input_data}
                  onChange={(e) =>
                    setFormData({ ...formData, input_data: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
                  placeholder="예: Boston House"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">
                  카테고리
                </label>
                <select
                  value={formData.category}
                  onChange={(e) =>
                    setFormData({ ...formData, category: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
                >
                  <option value="머신러닝">머신러닝</option>
                  <option value="딥러닝">딥러닝</option>
                  <option value="통계분석">통계분석</option>
                  <option value="DFA">DFA</option>
                  <option value="프라이싱">프라이싱</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-300 mb-1">설명</label>
                <textarea
                  value={formData.description}
                  onChange={(e) =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  className="w-full px-3 py-2 bg-gray-700 text-white rounded border border-gray-600 focus:border-purple-500 focus:outline-none"
                  rows={3}
                  placeholder="모델에 대한 설명을 입력하세요"
                />
              </div>
              <div className="flex gap-3 justify-end pt-4 mt-4 border-t border-gray-700">
                <button
                  type="button"
                  onClick={handleCancel}
                  disabled={loading}
                  className="px-5 py-2.5 bg-gray-600 text-white rounded-md hover:bg-gray-700 active:bg-gray-800 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed min-w-[80px]"
                >
                  취소
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={loading || !formData.name || formData.name.trim() === ""}
                  className="px-6 py-2.5 bg-purple-600 text-white rounded-md hover:bg-purple-700 active:bg-purple-800 transition-colors font-semibold shadow-lg shadow-purple-600/30 disabled:opacity-70 disabled:cursor-not-allowed flex items-center justify-center gap-2 min-w-[100px]"
                  style={{ display: 'flex' }}
                >
                  {loading ? (
                    <>
                      <svg
                        className="animate-spin h-4 w-4 text-white"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle
                          className="opacity-25"
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="4"
                        ></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                        ></path>
                      </svg>
                      <span>저장 중...</span>
                    </>
                  ) : (
                    <span>저장</span>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
