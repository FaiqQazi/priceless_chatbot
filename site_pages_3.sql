CREATE OR REPLACE FUNCTION public.filter_experiences(filter_data JSONB)
RETURNS SETOF site_pages AS $$
BEGIN
  RETURN QUERY 
  SELECT * FROM site_pages
  WHERE 
    -- Convert text deadline_date to date and filter
    (filter_data->>'deadline_date_gte')::DATE <= (deadline_date::DATE)

    -- Location matching: Check if any location in the DB contains a location from the filter
    AND EXISTS (
      SELECT 1 
      FROM jsonb_array_elements_text(filter_data->'location_filter') AS loc
      WHERE site_pages.location ILIKE '%' || loc || '%'
    );
END;
$$ LANGUAGE plpgsql;
