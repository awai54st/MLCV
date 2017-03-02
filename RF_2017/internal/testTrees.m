function label = testTrees(data,tree,param)
% Slow version - pass data point one-by-one

cc = [];
[N,D] = size(data);
len_T = length(tree); len_m = size(data,1);
label = [];
parfor T = 1:length(tree)
    label_tmp = zeros(len_m,1);
    for m = 1:size(data,1);
        idx = 1;
        
        while isempty(tree(T).node(idx).leaf_idx)
            t = tree(T).node(idx).t;
            dim = tree(T).node(idx).dim;
            % Decision
%             if data(m,dim) < t % Pass data to left node
            split_func = param.split_func;
            if split_func==1 
                decision = data(m,dim) < t;
            elseif split_func==2
                decision = [data(m,1:D-1),1]*t' > 0;
            elseif split_func==3
                data_hd = [data(m,1:D-1),1,data(m,dim(1)).^2,data(m,dim(2)).^2,data(m,dim(1)).*data(m,dim(2))];
                decision = (data_hd(m,:)*t') > 0;
            elseif split_func==4
                diff = data(m,dim(1))-data(m,dim(2));
                decision = ([diff,1]*t') > 0;
            end
            
            idxL = idx*2; idxR = idx*2+1;
            if isempty(tree(T).node(idxR).leaf_idx) && all(~tree(T).node(idxR).dim) % Empty right branch
                idx = idxL;
            elseif all(~tree(T).node(idxL).dim) && isempty(tree(T).node(idxL).leaf_idx) % Empty left branch
                idx = idxR;
            elseif all(~tree(T).node(idxL).dim) && all(~tree(T).node(idxR).dim) && isempty(tree(T).node(idxL).leaf_idx) && isempty(tree(T).node(idxR).leaf_idx)
                error('Error: No children nodes && not a leaf.');
            else
                if decision % Pass data to left node
                    idx = idxL;
                else
                    idx = idxR; % and to right
                end
            end
            
%             if decision % Pass data to left node
%                 idx = idx*2;
%             else
%                 idx = idx*2+1; % and to right
%             end
            
        end
        leaf_idx = tree(T).node(idx).leaf_idx;
        if ~isempty(tree(T).leaf(leaf_idx))
%             p(m,:,T) = tree(T).leaf(leaf_idx).prob;
%             label_tmp = [label_tmp;tree(T).leaf(leaf_idx).label];
            label_tmp(m) = tree(T).leaf(leaf_idx).label;
%             label(m,T) = tree(T).leaf(leaf_idx).label;
%             if isfield(tree(T).leaf(leaf_idx),'cc') % for clustering forest
%                 cc(m,:,T) = tree(T).leaf(leaf_idx).cc;
%             end
        end
        
    end
    label = [label,label_tmp];
end

end

