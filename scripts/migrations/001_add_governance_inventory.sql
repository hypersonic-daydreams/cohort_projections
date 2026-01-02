CREATE TABLE IF NOT EXISTS governance_inventory (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filepath TEXT UNIQUE NOT NULL,
    doc_type TEXT NOT NULL, -- 'ADR', 'SOP', 'REPORT', 'TEMPLATE', 'OTHER'
    doc_id TEXT, -- '001', '022', etc.
    title TEXT,
    status TEXT, -- 'ACCEPTED', 'PROPOSED', 'DEPRECATED', 'DRAFT', 'UNKNOWN'
    last_reviewed_date DATE,
    next_review_due DATE,
    metadata JSONB DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast status lookups
CREATE INDEX IF NOT EXISTS idx_governance_status ON governance_inventory(status);
CREATE INDEX IF NOT EXISTS idx_governance_doc_type ON governance_inventory(doc_type);
