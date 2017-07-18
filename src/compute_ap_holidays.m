function AP = compute_ap_holidays(simi, db)

nQueries = numel(db.q_idx);
if size(simi, 1) == size(simi, 2)
    simi = simi(:, db.q_idx);
end
AP = zeros(1, nQueries);
for q = 1:nQueries
    posi_list = db.posIDs{q};
    junk_list = db.ignoreIDs{q};
    
    simi_i = simi(:,q);

    [~, ranked_list] = sort(-simi_i);
    n = length(simi_i);
    l = length(posi_list);

    old_recall    = 0;
    old_precision = 1;
    ap = 0;

    j=1;
    intersect_size = 0;
    for i = 1:n
        idx = ranked_list(i);
        if ~(ismember(idx, junk_list))
            if ismember(idx, posi_list)
                intersect_size = intersect_size+1;
            end
            recall = intersect_size/l;
            precision = intersect_size/j;
        
            ap = ap + (recall-old_recall)*(precision+old_precision)/2;
        
            old_recall = recall;
            old_precision = precision;
            j = j+1;
        end
    end
    AP(q) = ap;
end