-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table for storing processed chunks of site pages
CREATE TABLE site_pages (
    id BIGSERIAL PRIMARY KEY,                        -- Unique identifier for each chunk
    url VARCHAR NOT NULL,                            -- URL of the processed site
    title TEXT NOT NULL,                             -- Title of the chunk
    summary TEXT NOT NULL,                           -- Summary of the chunk
    content TEXT NOT NULL,                           -- Full text content of the chunk
    metadata JSONB NOT NULL DEFAULT '{}'::JSONB,     -- Metadata as JSON, includes additional details
    embedding VECTOR(1536),                          -- OpenAI embeddings with 1536 dimensions
    date TEXT NOT NULL DEFAULT 'Unknown',            -- Date associated with the chunk
    category TEXT NOT NULL DEFAULT 'Unknown',        -- Category of the content
    location TEXT NOT NULL DEFAULT 'Unknown',        -- Location associated with the chunk
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()) NOT NULL,  -- Creation timestamp

    -- Unique constraint to prevent duplicate chunks for the same URL and title
    UNIQUE (url, title)
);

-- Index for vector similarity search
CREATE INDEX ON site_pages USING ivfflat (embedding vector_cosine_ops);

-- Index on metadata for faster filtering and querying
CREATE INDEX idx_site_pages_metadata ON site_pages USING gin (metadata);

-- Function to perform similarity-based vector searches with optional filtering
CREATE FUNCTION match_site_pages (
    query_embedding VECTOR(1536),                   -- Input embedding to match against
    match_count INT DEFAULT 10,                     -- Number of results to return
    filter JSONB DEFAULT '{}'::JSONB                -- Optional filter for metadata-based queries
) RETURNS TABLE (
    id BIGINT,
    url VARCHAR,
    title TEXT,
    summary TEXT,
    content TEXT,
    metadata JSONB,
    date TEXT,
    category TEXT,
    location TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        site_pages.id,  -- Explicitly qualify "id" to avoid ambiguity
        site_pages.url,
        site_pages.title,
        site_pages.summary,
        site_pages.content,
        site_pages.metadata,
        site_pages.date,
        site_pages.category,
        site_pages.location,
        1 - (site_pages.embedding <=> query_embedding) AS similarity
    FROM site_pages  -- Match metadata if the filter is provided
    ORDER BY site_pages.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;


-- Enable row-level security (RLS) on the table
ALTER TABLE site_pages ENABLE ROW LEVEL SECURITY;

-- Create a policy to allow public read access
CREATE POLICY "Allow public read access"
    ON site_pages
    FOR SELECT
    TO public
    USING (true);
