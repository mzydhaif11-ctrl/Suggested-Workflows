// src/extensions/prompt-studio/types.ts

export type PromptCategory = 'creative' | 'analytical' | 'coding' | 'general';

export interface PromptVariable {
  key: string;
  label: string;
  defaultValue: string;
}

export interface PromptTemplate {
  id: string;
  title: string;
  category: PromptCategory;
  templateText: string;
  variables: PromptVariable[];
}

export interface PromptStudioPayload {
  rawIdea: string;
  enhancedPrompt: string;
  selectedTemplateId?: string;
  variables: Record<string, string>;
  parameters: {
    temperature: number;
    maxTokens: number;
  };
}
