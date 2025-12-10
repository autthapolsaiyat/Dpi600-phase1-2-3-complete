-- ============================================================================
-- DPI-600 DRUG PROFILE INTELLIGENCE SYSTEM
-- Database Schema Extension: Pill Fingerprint Profiles
-- Version: 1.0
-- Created: December 2025
-- Author: Autthapol Saiyat (Boy)
-- ============================================================================

-- ============================================================================
-- SECTION 1: DRUG PILL PROFILES (ลายนิ้วมือเม็ดยา)
-- ============================================================================

-- Main Pill Profile table - stores unique "fingerprints" of drug pills
CREATE TABLE pill_profiles (
    id SERIAL PRIMARY KEY,
    profile_id VARCHAR(30) UNIQUE NOT NULL,     -- e.g., 'DPF-2568-0042'
    
    -- Visual Signature
    logo_type VARCHAR(50),                       -- lion, wy, 999, etc.
    logo_confidence DECIMAL(5,2),
    shape VARCHAR(30),                           -- round, oval, rectangle, etc.
    
    -- Physical Dimensions (in mm)
    diameter DECIMAL(6,2),
    diameter_tolerance DECIMAL(4,2) DEFAULT 0.1, -- ±0.1mm
    thickness DECIMAL(6,2),
    thickness_tolerance DECIMAL(4,2) DEFAULT 0.05,
    weight_mg DECIMAL(8,2),                      -- weight in milligrams
    
    -- Color Analysis
    color_primary VARCHAR(7),                    -- hex color e.g., '#FF6B35'
    color_secondary VARCHAR(7),
    color_histogram JSONB,                       -- detailed color distribution
    color_delta_e_tolerance DECIMAL(4,2) DEFAULT 5.0, -- color difference tolerance
    
    -- Surface Features
    surface_type VARCHAR(30),                    -- smooth, textured, scored
    edge_type VARCHAR(30),                       -- beveled, flat, rounded
    emboss_depth DECIMAL(4,2),                   -- depth of embossed logo
    back_marking VARCHAR(100),                   -- score line, blank, etc.
    
    -- Chemical Signature
    chemical_profile JSONB NOT NULL,             -- {"meth": 85, "caff": 40, ...}
    chemical_tolerance DECIMAL(4,2) DEFAULT 5.0, -- ±5% tolerance
    trace_elements JSONB,                        -- trace elements detected
    
    -- AI Model Reference
    shape_vector FLOAT[],                        -- shape descriptor vector (Hu Moments)
    color_vector FLOAT[],                        -- color histogram vector
    texture_vector FLOAT[],                      -- texture descriptor (GLCM)
    
    -- Reference Image
    reference_image_url TEXT,
    reference_image_hash VARCHAR(64),            -- SHA256 hash for deduplication
    
    -- Statistics
    total_matched_cases INTEGER DEFAULT 0,
    total_pills_seized INTEGER DEFAULT 0,
    first_seen_date DATE,
    last_seen_date DATE,
    primary_regions TEXT[],
    
    -- Linked Organization
    organization_id INTEGER REFERENCES organizations(id),
    org_confidence DECIMAL(5,2),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',         -- active, merged, archived
    confidence_score DECIMAL(5,2),               -- overall profile confidence
    verified BOOLEAN DEFAULT FALSE,
    verified_by INTEGER,
    verified_at TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    created_by INTEGER,
    notes TEXT
);

-- ============================================================================
-- SECTION 2: PILL PROFILE IMAGES
-- ============================================================================

-- Reference images for each profile
CREATE TABLE pill_profile_images (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER REFERENCES pill_profiles(id) ON DELETE CASCADE,
    
    image_url TEXT NOT NULL,
    image_type VARCHAR(30),                      -- reference, sample, processed
    image_hash VARCHAR(64),                      -- SHA256 for deduplication
    
    -- Image metadata
    capture_date TIMESTAMP,
    capture_device VARCHAR(100),
    image_quality_score DECIMAL(5,2),
    
    -- Analysis results
    ai_analyzed BOOLEAN DEFAULT FALSE,
    shape_vector FLOAT[],
    color_vector FLOAT[],
    
    -- Source
    source_case_id INTEGER REFERENCES drug_cases(id),
    source_lab_id VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- SECTION 3: CASE-PROFILE MATCHING
-- ============================================================================

-- Links cases to pill profiles
CREATE TABLE case_profile_matches (
    id SERIAL PRIMARY KEY,
    case_id INTEGER REFERENCES drug_cases(id) ON DELETE CASCADE,
    profile_id INTEGER REFERENCES pill_profiles(id) ON DELETE CASCADE,
    
    -- Match Scores (0-100)
    visual_match_score DECIMAL(5,2),             -- shape + color + size
    shape_match_score DECIMAL(5,2),
    color_match_score DECIMAL(5,2),
    dimension_match_score DECIMAL(5,2),
    chemical_match_score DECIMAL(5,2),
    overall_match_score DECIMAL(5,2),
    
    -- Match Details
    dimension_diffs JSONB,                       -- {"diameter": -1.6, "thickness": +3.3}
    chemical_diffs JSONB,                        -- {"meth": -1, "caff": +2}
    
    -- Within Tolerance Check
    within_visual_tolerance BOOLEAN,
    within_chemical_tolerance BOOLEAN,
    is_same_batch BOOLEAN,                       -- confirmed same production batch
    
    -- Verification
    verified BOOLEAN DEFAULT FALSE,
    verified_by INTEGER,
    verified_at TIMESTAMP,
    verification_notes TEXT,
    
    -- Metadata
    matched_at TIMESTAMP DEFAULT NOW(),
    match_algorithm_version VARCHAR(20) DEFAULT '1.0'
);

-- ============================================================================
-- SECTION 4: PROFILE COMPARISON HISTORY
-- ============================================================================

-- Logs all comparisons for audit trail
CREATE TABLE profile_comparison_log (
    id SERIAL PRIMARY KEY,
    
    -- What was compared
    sample_case_id INTEGER REFERENCES drug_cases(id),
    sample_image_url TEXT,
    reference_profile_id INTEGER REFERENCES pill_profiles(id),
    
    -- Comparison Results
    visual_score DECIMAL(5,2),
    chemical_score DECIMAL(5,2),
    overall_score DECIMAL(5,2),
    
    -- Outcome
    result VARCHAR(30),                          -- matched, no_match, possible_match
    action_taken VARCHAR(50),                    -- created_profile, added_to_profile, rejected
    
    -- Who did it
    compared_by INTEGER REFERENCES users(id),
    compared_at TIMESTAMP DEFAULT NOW(),
    
    -- Additional data
    comparison_params JSONB,                     -- tolerance settings used
    notes TEXT
);

-- ============================================================================
-- SECTION 5: BATCH TRACKING
-- ============================================================================

-- Track pills believed to be from same production batch
CREATE TABLE production_batches (
    id SERIAL PRIMARY KEY,
    batch_code VARCHAR(30) UNIQUE NOT NULL,      -- e.g., 'BATCH-2568-LION-001'
    
    -- Primary Profile
    primary_profile_id INTEGER REFERENCES pill_profiles(id),
    
    -- Batch Characteristics
    estimated_production_date DATE,
    estimated_quantity INTEGER,
    
    -- Geographic Spread
    first_seizure_date DATE,
    first_seizure_province VARCHAR(100),
    last_seizure_date DATE,
    geographic_spread TEXT[],                    -- provinces where found
    
    -- Statistics
    total_seizures INTEGER DEFAULT 0,
    total_pills INTEGER DEFAULT 0,
    
    -- Linked Organization
    organization_id INTEGER REFERENCES organizations(id),
    
    -- Status
    status VARCHAR(20) DEFAULT 'active',         -- active, depleted, tracking
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Link cases to batches
CREATE TABLE batch_cases (
    id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES production_batches(id) ON DELETE CASCADE,
    case_id INTEGER REFERENCES drug_cases(id) ON DELETE CASCADE,
    match_confidence DECIMAL(5,2),
    added_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(batch_id, case_id)
);

-- ============================================================================
-- SECTION 6: IMAGE STANDARDS (from พิสูจน์หลักฐาน)
-- ============================================================================

-- Standard imaging requirements
CREATE TABLE imaging_standards (
    id SERIAL PRIMARY KEY,
    standard_code VARCHAR(30) UNIQUE NOT NULL,
    name VARCHAR(200),
    
    -- Camera Requirements
    camera_angle_degrees INTEGER,                -- e.g., 90 (top-down)
    min_resolution_mp DECIMAL(4,1),              -- megapixels
    lighting_type VARCHAR(50),                   -- diffused, ring light, etc.
    
    -- Background
    background_color VARCHAR(7),                 -- hex color
    background_type VARCHAR(50),                 -- matte, glossy, etc.
    
    -- Scale Reference
    scale_required BOOLEAN DEFAULT TRUE,
    scale_type VARCHAR(50),                      -- ruler, reference card, etc.
    scale_unit VARCHAR(10),                      -- mm, cm
    
    -- Image Format
    file_format VARCHAR(10),                     -- PNG, TIFF, etc.
    color_space VARCHAR(20),                     -- sRGB, Adobe RGB
    
    -- Quality Metrics
    min_sharpness_score DECIMAL(5,2),
    max_noise_level DECIMAL(5,2),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    notes TEXT
);

-- Insert default standard (from พิสูจน์หลักฐาน)
INSERT INTO imaging_standards (
    standard_code, name, camera_angle_degrees, min_resolution_mp,
    lighting_type, background_color, scale_required, scale_type,
    file_format, color_space
) VALUES (
    'STD-FORENSIC-001',
    'มาตรฐานภาพถ่ายยาเสพติด - สำนักงานพิสูจน์หลักฐานตำรวจ',
    90, 12.0,
    'diffused ring light', '#FFFFFF', TRUE, 'forensic scale card',
    'PNG', 'sRGB'
);

-- ============================================================================
-- SECTION 7: INDEXES
-- ============================================================================

-- Pill Profiles
CREATE INDEX idx_profile_logo ON pill_profiles(logo_type);
CREATE INDEX idx_profile_shape ON pill_profiles(shape);
CREATE INDEX idx_profile_org ON pill_profiles(organization_id);
CREATE INDEX idx_profile_status ON pill_profiles(status);
CREATE INDEX idx_profile_diameter ON pill_profiles(diameter);
CREATE INDEX idx_profile_created ON pill_profiles(created_at DESC);

-- Case-Profile Matches
CREATE INDEX idx_cpm_case ON case_profile_matches(case_id);
CREATE INDEX idx_cpm_profile ON case_profile_matches(profile_id);
CREATE INDEX idx_cpm_score ON case_profile_matches(overall_match_score DESC);
CREATE INDEX idx_cpm_batch ON case_profile_matches(is_same_batch);

-- Batches
CREATE INDEX idx_batch_profile ON production_batches(primary_profile_id);
CREATE INDEX idx_batch_org ON production_batches(organization_id);

-- ============================================================================
-- SECTION 8: FUNCTIONS
-- ============================================================================

-- Calculate visual similarity between two pill images
CREATE OR REPLACE FUNCTION calculate_visual_similarity(
    sample_dimensions JSONB,
    ref_dimensions JSONB,
    tolerance DECIMAL DEFAULT 5.0
) RETURNS JSONB AS $$
DECLARE
    diameter_diff DECIMAL;
    thickness_diff DECIMAL;
    shape_score DECIMAL;
    within_tolerance BOOLEAN;
BEGIN
    -- Calculate percentage differences
    diameter_diff := ABS(
        ((sample_dimensions->>'diameter')::DECIMAL - (ref_dimensions->>'diameter')::DECIMAL) 
        / (ref_dimensions->>'diameter')::DECIMAL * 100
    );
    
    thickness_diff := ABS(
        ((sample_dimensions->>'thickness')::DECIMAL - (ref_dimensions->>'thickness')::DECIMAL) 
        / (ref_dimensions->>'thickness')::DECIMAL * 100
    );
    
    -- Calculate shape score
    shape_score := 100 - (diameter_diff + thickness_diff) / 2;
    
    -- Check if within tolerance
    within_tolerance := (diameter_diff <= tolerance AND thickness_diff <= tolerance);
    
    RETURN jsonb_build_object(
        'diameter_diff_pct', ROUND(diameter_diff, 2),
        'thickness_diff_pct', ROUND(thickness_diff, 2),
        'shape_score', ROUND(shape_score, 2),
        'within_tolerance', within_tolerance
    );
END;
$$ LANGUAGE plpgsql;

-- Calculate overall pill match score
CREATE OR REPLACE FUNCTION calculate_pill_match_score(
    visual_score DECIMAL,
    chemical_score DECIMAL,
    visual_weight DECIMAL DEFAULT 0.5,
    chemical_weight DECIMAL DEFAULT 0.5
) RETURNS DECIMAL(5,2) AS $$
BEGIN
    RETURN ROUND(visual_score * visual_weight + chemical_score * chemical_weight, 2);
END;
$$ LANGUAGE plpgsql;

-- Check if match is within tolerance
CREATE OR REPLACE FUNCTION is_within_tolerance(
    diff_pct DECIMAL,
    tolerance DECIMAL DEFAULT 5.0
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN ABS(diff_pct) <= tolerance;
END;
$$ LANGUAGE plpgsql;

-- Generate Profile ID
CREATE OR REPLACE FUNCTION generate_profile_id() 
RETURNS VARCHAR(30) AS $$
DECLARE
    year_be INTEGER;
    seq_num INTEGER;
BEGIN
    year_be := EXTRACT(YEAR FROM NOW()) + 543;
    
    SELECT COALESCE(MAX(
        CAST(SUBSTRING(profile_id FROM 10 FOR 4) AS INTEGER)
    ), 0) + 1
    INTO seq_num
    FROM pill_profiles
    WHERE profile_id LIKE 'DPF-' || year_be || '-%';
    
    RETURN 'DPF-' || year_be || '-' || LPAD(seq_num::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SECTION 9: VIEWS
-- ============================================================================

-- Profile Summary with Statistics
CREATE OR REPLACE VIEW v_profile_summary AS
SELECT 
    p.id,
    p.profile_id,
    p.logo_type,
    p.shape,
    p.diameter,
    p.thickness,
    p.chemical_profile,
    o.name AS organization_name,
    p.org_confidence,
    p.total_matched_cases,
    p.total_pills_seized,
    p.first_seen_date,
    p.last_seen_date,
    p.primary_regions,
    p.status,
    p.verified
FROM pill_profiles p
LEFT JOIN organizations o ON p.organization_id = o.id
ORDER BY p.total_matched_cases DESC;

-- Cases matching each profile
CREATE OR REPLACE VIEW v_profile_cases AS
SELECT 
    p.profile_id,
    p.logo_type,
    c.case_number,
    c.case_date,
    c.province,
    c.quantity_pills,
    m.overall_match_score,
    m.is_same_batch
FROM pill_profiles p
JOIN case_profile_matches m ON p.id = m.profile_id
JOIN drug_cases c ON m.case_id = c.id
ORDER BY p.profile_id, m.overall_match_score DESC;

-- Batch Timeline
CREATE OR REPLACE VIEW v_batch_timeline AS
SELECT 
    b.batch_code,
    b.estimated_production_date,
    c.case_number,
    c.case_date,
    c.province,
    c.quantity_pills,
    bc.match_confidence
FROM production_batches b
JOIN batch_cases bc ON b.id = bc.batch_id
JOIN drug_cases c ON bc.case_id = c.id
ORDER BY b.batch_code, c.case_date;

-- ============================================================================
-- SECTION 10: SAMPLE DATA
-- ============================================================================

-- Insert sample pill profiles
INSERT INTO pill_profiles (
    profile_id, logo_type, logo_confidence, shape,
    diameter, thickness, color_primary,
    chemical_profile, chemical_tolerance,
    organization_id, org_confidence,
    total_matched_cases, total_pills_seized,
    first_seen_date, last_seen_date,
    primary_regions, status, verified
) VALUES 
(
    'DPF-2568-0001', 'lion', 97.5, 'round',
    6.2, 3.1, '#FF6B35',
    '{"meth": 85, "caff": 40, "lact": 20, "mdma": 5}', 5.0,
    1, 97.5,
    47, 2350000,
    '2566-01-15', '2568-12-01',
    ARRAY['เชียงราย', 'เชียงใหม่', 'แม่ฮ่องสอน'], 'active', TRUE
),
(
    'DPF-2568-0002', 'wy', 92.3, 'round',
    5.8, 2.9, '#44AAFF',
    '{"meth": 78, "caff": 55, "lact": 25, "mdma": 8}', 5.0,
    2, 91.5,
    23, 890000,
    '2567-03-20', '2568-11-28',
    ARRAY['นครพนม', 'มุกดาหาร', 'อุบลราชธานี'], 'active', TRUE
),
(
    'DPF-2568-0003', '999', 88.7, 'round',
    6.0, 3.2, '#FFCC00',
    '{"meth": 70, "caff": 30, "lact": 35, "mdma": 15}', 5.0,
    3, 89.2,
    12, 450000,
    '2568-02-10', '2568-11-15',
    ARRAY['กรุงเทพมหานคร', 'สมุทรปราการ'], 'active', TRUE
),
(
    'DPF-2568-0004', 'horse', 90.1, 'oval',
    6.5, 3.1, '#88FF44',
    '{"meth": 82, "caff": 35, "lact": 28, "mdma": 3}', 5.0,
    4, 88.7,
    18, 620000,
    '2566-08-05', '2568-10-20',
    ARRAY['กาญจนบุรี', 'ตาก', 'ราชบุรี'], 'active', TRUE
);

-- Insert sample case-profile matches
INSERT INTO case_profile_matches (
    case_id, profile_id, 
    visual_match_score, shape_match_score, color_match_score, 
    dimension_match_score, chemical_match_score, overall_match_score,
    dimension_diffs, chemical_diffs,
    within_visual_tolerance, within_chemical_tolerance, is_same_batch
) VALUES 
(1, 1, 96.8, 97.2, 95.5, 98.4, 97.2, 97.0, 
 '{"diameter": -1.6, "thickness": 3.3}', '{"meth": -1, "caff": 2, "lact": -1, "mdma": 1}',
 TRUE, TRUE, TRUE),
(4, 1, 95.2, 96.0, 94.8, 97.1, 96.8, 96.0,
 '{"diameter": -2.1, "thickness": 2.8}', '{"meth": -2, "caff": 1, "lact": -2, "mdma": 1}',
 TRUE, TRUE, TRUE),
(2, 2, 94.5, 95.2, 93.8, 96.2, 95.1, 94.8,
 '{"diameter": -1.8, "thickness": 3.1}', '{"meth": 1, "caff": -1, "lact": -1, "mdma": -1}',
 TRUE, TRUE, TRUE),
(5, 2, 93.2, 94.0, 92.5, 95.8, 94.2, 93.7,
 '{"diameter": -2.3, "thickness": 2.5}', '{"meth": -1, "caff": 1, "lact": 1, "mdma": 1}',
 TRUE, TRUE, TRUE),
(3, 3, 91.5, 92.3, 90.8, 94.1, 93.5, 92.5,
 '{"diameter": -1.5, "thickness": 2.9}', '{"meth": 1, "caff": 2, "lact": -2, "mdma": -1}',
 TRUE, TRUE, TRUE);

-- Insert sample production batch
INSERT INTO production_batches (
    batch_code, primary_profile_id, 
    estimated_production_date, estimated_quantity,
    first_seizure_date, first_seizure_province,
    last_seizure_date, geographic_spread,
    total_seizures, total_pills, organization_id
) VALUES (
    'BATCH-2568-LION-001', 1,
    '2568-10-15', 500000,
    '2568-11-01', 'เชียงราย',
    '2568-12-01', ARRAY['เชียงราย', 'เชียงใหม่', 'ลำปาง'],
    5, 185000, 1
);

-- Link cases to batch
INSERT INTO batch_cases (batch_id, case_id, match_confidence) VALUES
(1, 1, 97.0),
(1, 4, 96.0);

-- ============================================================================
-- SECTION 11: TRIGGERS
-- ============================================================================

-- Auto-update pill profile statistics
CREATE OR REPLACE FUNCTION update_profile_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE pill_profiles
    SET 
        total_matched_cases = (
            SELECT COUNT(*) FROM case_profile_matches 
            WHERE profile_id = NEW.profile_id AND is_same_batch = TRUE
        ),
        last_seen_date = (
            SELECT MAX(c.case_date) FROM case_profile_matches m
            JOIN drug_cases c ON m.case_id = c.id
            WHERE m.profile_id = NEW.profile_id
        ),
        updated_at = NOW()
    WHERE id = NEW.profile_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_profile_stats
AFTER INSERT OR UPDATE ON case_profile_matches
FOR EACH ROW EXECUTE FUNCTION update_profile_stats();

-- Auto-update batch statistics
CREATE OR REPLACE FUNCTION update_batch_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE production_batches
    SET 
        total_seizures = (
            SELECT COUNT(*) FROM batch_cases WHERE batch_id = NEW.batch_id
        ),
        total_pills = (
            SELECT COALESCE(SUM(c.quantity_pills), 0) 
            FROM batch_cases bc
            JOIN drug_cases c ON bc.case_id = c.id
            WHERE bc.batch_id = NEW.batch_id
        ),
        last_seizure_date = (
            SELECT MAX(c.case_date) 
            FROM batch_cases bc
            JOIN drug_cases c ON bc.case_id = c.id
            WHERE bc.batch_id = NEW.batch_id
        ),
        updated_at = NOW()
    WHERE id = NEW.batch_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tr_update_batch_stats
AFTER INSERT OR UPDATE OR DELETE ON batch_cases
FOR EACH ROW EXECUTE FUNCTION update_batch_stats();

-- ============================================================================
-- END OF SCHEMA EXTENSION
-- ============================================================================
