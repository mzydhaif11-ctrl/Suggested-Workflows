// src/extensions/prompt-studio/index.ts

import { PromptStudio } from './PromptStudio';
import { PromptStudioPayload } from './types';

export const PromptStudioExtension = {
  id: 'extension-prompt-studio',
  title: 'فكرة الموجّه',
  slot: 'sidebar',
  component: PromptStudio
};

// دالة تحديث عقدة سير العمل النشطة
export const applyPromptToWorkflowNode = (
  payload: PromptStudioPayload,
  updateNode: (nodeId: string, data: Record<string, any>) => void,
  targetNodeId?: string
) => {
  if (!targetNodeId) {
    console.warn('لم يتم تحديد عقدة نشطة في سير العمل.');
    return;
  }
  updateNode(targetNodeId, {
    prompt: payload.enhancedPrompt,
    temperature: payload.parameters.temperature,
    metadata: {
      rawIdea: payload.rawIdea,
      templateId: payload.selectedTemplateId
    }
  });
};
