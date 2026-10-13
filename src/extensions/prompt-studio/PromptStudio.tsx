// src/extensions/prompt-studio/PromptStudio.tsx

import React, { useState } from 'react';
import { PromptStudioPayload, PromptTemplate } from './types';

const DEFAULT_TEMPLATES: PromptTemplate[] = [
  {
    id: 'expert-advisor',
    title: 'خبير استشاري',
    category: 'analytical',
    templateText: 'تصرف كخبير في {{field}}. حلل مسألة {{topic}} وقدم دليلاً عملياً.',
    variables: [
      { key: 'field', label: 'المجال', defaultValue: 'تطوير البرمجيات' },
      { key: 'topic', label: 'الموضوع', defaultValue: 'أتمتة الأعمال' }
    ]
  },
  {
    id: 'creative-prompt',
    title: 'توليد إبداعي',
    category: 'creative',
    templateText: 'ابتكر فكرة جديدة حول {{concept}} مع مراعاة الجمهور: {{audience}}.',
    variables: [
      { key: 'concept', label: 'الفكرة', defaultValue: 'أداة ذكاء اصطناعي' },
      { key: 'audience', label: 'المستهدفون', defaultValue: 'المصممون' }
    ]
  }
];

interface PromptStudioProps {
  onInjectIntoWorkflow: (payload: PromptStudioPayload) => void;
}

export const PromptStudio: React.FC<PromptStudioProps> = ({ onInjectIntoWorkflow }) => {
  const [rawIdea, setRawIdea] = useState('');
  const [enhancedPrompt, setEnhancedPrompt] = useState('');
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTemplate, setActiveTemplate] = useState<PromptTemplate | null>(null);
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [temperature, setTemperature] = useState(0.7);

  // إرسال الفكرة إلى OpenAI عبر الـ Backend
  const handleEnhance = async () => {
    if (!rawIdea.trim()) return;
    setIsLoading(true);
    try {
      const response = await fetch('/api/prompt-studio/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          rawIdea, 
          model: selectedModel,
          temperature 
        })
      });
      const data = await response.json();
      if (response.ok) {
        setEnhancedPrompt(data.enhancedPrompt);
      } else {
        alert(`خطأ: ${data.detail || 'تعذر التحسين'}`);
      }
    } catch (err) {
      console.error('Error enhancing prompt:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectTemplate = (template: PromptTemplate) => {
    setActiveTemplate(template);
    const initialVars: Record<string, string> = {};
    template.variables.forEach(v => {
      initialVars[v.key] = v.defaultValue;
    });
    setVariables(initialVars);
    setEnhancedPrompt(template.templateText);
  };

  const handleApply = () => {
    let finalPrompt = enhancedPrompt || rawIdea;
    Object.entries(variables).forEach(([k, v]) => {
      finalPrompt = finalPrompt.replace(new RegExp(`{{${k}}}`, 'g'), v);
    });

    onInjectIntoWorkflow({
      rawIdea,
      enhancedPrompt: finalPrompt,
      selectedTemplateId: activeTemplate?.id,
      variables,
      parameters: { temperature, maxTokens: 2048 }
    });
  };

  return (
    <div className="flex flex-col gap-4 p-4 bg-slate-900 text-slate-100 rounded-xl border border-slate-800 w-full shadow-lg">
      <div className="border-b border-slate-800 pb-2 flex justify-between items-center">
        <div>
          <h3 className="text-base font-bold text-slate-100">واجهة فكرة الموجّه (OpenAI)</h3>
          <p className="text-xs text-slate-400">صياغة وتوسيع الأفكار باستخدام نماذج GPT</p>
        </div>
        {/* اختيار النموذج */}
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="text-xs bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-slate-200 focus:outline-none focus:border-blue-500"
        >
          <option value="gpt-4o-mini">gpt-4o-mini (سريع)</option>
          <option value="gpt-4o">gpt-4o (متقدم)</option>
        </select>
      </div>

      {/* حقل الفكرة */}
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold text-slate-300">الفكرة الأولية:</label>
        <textarea
          rows={3}
          value={rawIdea}
          onChange={(e) => setRawIdea(e.target.value)}
          placeholder="اكتب فكرتك باختصار هنا..."
          className="w-full text-xs p-2.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={handleEnhance}
          disabled={isLoading || !rawIdea.trim()}
          className="self-start text-xs font-medium px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-md transition"
        >
          {isLoading ? 'جاري التحسين عبر OpenAI...' : '✨ تحسين الفكرة عبر OpenAI'}
        </button>
      </div>

      {/* قوالب سريعة */}
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold text-slate-300">قوالب مقترحة:</label>
        <div className="flex gap-2 flex-wrap">
          {DEFAULT_TEMPLATES.map((tmpl) => (
            <button
              key={tmpl.id}
              onClick={() => handleSelectTemplate(tmpl)}
              className={`text-xs px-2.5 py-1 rounded-md border transition ${
                activeTemplate?.id === tmpl.id
                  ? 'border-emerald-500 bg-emerald-950 text-emerald-200'
                  : 'border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {tmpl.title}
            </button>
          ))}
        </div>
      </div>

      {/* محرر الموجه المحسن الناتج من OpenAI */}
      <div className="flex flex-col gap-1.5">
        <label className="text-xs font-semibold text-slate-300">الموجّه النهائي الجاهز:</label>
        <textarea
          rows={4}
          value={enhancedPrompt}
          onChange={(e) => setEnhancedPrompt(e.target.value)}
          placeholder="سيظهر الموجه المطور هنا..."
          className="w-full text-xs p-2.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
        />
      </div>

      {/* زر التصدير للـ Workflow */}
      <button
        onClick={handleApply}
        disabled={!enhancedPrompt.trim() && !rawIdea.trim()}
        className="w-full py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-medium text-xs rounded-lg transition"
      >
        اعتماد الموجه وإرساله إلى تدفق العمل (Workflow)
      </button>
    </div>
  );
};
