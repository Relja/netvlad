% Author: Relja Arandjelovic (relja@relja.info)

function db= dbVGG( dsetName, useROI )
    % useROI: negative: use full image; 0+: the amount to expand ROI with from each side
    % Expand by up to half the size of the receptive field
    
    if nargin<2, useROI= 0; end
    if nargin<1, dsetName='ox5k'; end
    
    db.name= dsetName;
    
    paths= localPaths();
    
    if strcmp(db.name,'ox5k')
        
        dsetRootOxford= relja_expandUser(paths.dsetRootOxford);
        dsetRoot= dsetRootOxford;
        db.dbPath= [dsetRootOxford, 'images/'];
        db.qPath= db.dbPath;
        
        db.dbImageFns= relja_dir(db.dbPath);
        assert( length(db.dbImageFns)== 5063 );
       
        isOx= true;

    elseif strcmp(db.name,'paris')
        
        dsetRootParis= relja_expandUser(paths.dsetRootParis);
        dsetRoot= dsetRootParis;
        db.dbPath= [dsetRootParis, 'images/'];
        db.qPath= db.dbPath;
        
        subdirs= relja_dir(db.dbPath);
        db.dbImageFns= {};
        for iDir= 1:length(subdirs)
            db.dbImageFns= [db.dbImageFns; strcat( [subdirs{iDir}, '/'], relja_dir( [db.dbPath, subdirs{iDir}] ) )];
        end
        assert( length(db.dbImageFns)== 6412 );
        
        corruptList= textread( sprintf('%scorrupt.txt', dsetRootParis), '%s');
        db.dbImageFns( ismember(db.dbImageFns, corruptList) )= [];
        assert( length(db.dbImageFns)== 6412-20 );
        
        isOx= false;
    else
        error( ['Not supported dataset: ', db.dsetName] );
    end
    
    db.numImages= length(db.dbImageFns);
    
    % get gt
    
    gtDir= sprintf('%sgroundtruth/', dsetRoot);
    dirlist= dir(gtDir);
    dirlist= sort({dirlist.name});
    
    % get queries
    
    db.numQueries= floor(length(dirlist)/4);
    assert(db.numQueries==55);
    db.queryNames= cell( db.numQueries, 1 );
    db.qImageFns= cell( db.numQueries, 1 );
    db.poss= cell( db.numQueries, 1 );
    db.ignores= cell( db.numQueries, 1 );
    db.posIDs= cell( db.numQueries, 1 );
    db.ignoreIDs= cell( db.numQueries, 1 );
    db.queryIDs= zeros( db.numQueries, 1 );
    db.ROIs= zeros(db.numQueries, 4);
    
    iQuery= 1;
    
    for i= 1:length(dirlist)
        if relja_endsWith( dirlist{i}, '_query.txt')
            db.queryNames{iQuery}= dirlist{i}(1:(end-10));
            iQuery= iQuery+1;
        end
    end
    
    if isOx
        onlyFns= db.dbImageFns;
    else
        % Paris
        onlyFns= db.dbImageFns;
        for iIm= 1:db.numImages
            [~, onlyFns{iIm}]= fileparts(onlyFns{iIm});
        end
        onlyFns= strcat(onlyFns, '.jpg');
    end
    
    % go one by one in sorted order
    
    for iQuery= 1:db.numQueries
        [qIm, db.ROIs(iQuery,1), db.ROIs(iQuery,2), db.ROIs(iQuery,3), db.ROIs(iQuery,4)]= ...
            textread( sprintf('%s%s_query.txt', gtDir, db.queryNames{iQuery}), '%s %f %f %f %f', 1);
        
        % read positives and ignores
        db.poss{iQuery}= textread(sprintf('%s%s_good.txt', gtDir, db.queryNames{iQuery}), '%s');
        db.poss{iQuery}= [db.poss{iQuery}; textread(sprintf('%s%s_ok.txt', gtDir, db.queryNames{iQuery}), '%s')];
        db.ignores{iQuery}= textread(sprintf('%s%s_junk.txt', gtDir, db.queryNames{iQuery}), '%s');
        
        % convert all to fns
        if isOx
            db.qImageFns{iQuery}= [qIm{1}(6:end),'.jpg'];
            qOnlyFn= db.qImageFns{iQuery};
        else
            qOnlyFn= [qIm{1},'.jpg'];
            db.qImageFns{iQuery}=  db.dbImageFns{ find(ismember(onlyFns, qOnlyFn)) };
            assert( relja_endsWith(db.qImageFns{iQuery}, qOnlyFn) ); % sanity check
        end
        db.poss{iQuery}= strcat(db.poss{iQuery}, '.jpg');
        db.ignores{iQuery}= strcat(db.ignores{iQuery}, '.jpg');

        % get IDs
        [tmp, db.queryIDs(iQuery)]= ismember(qOnlyFn, onlyFns);
        [tmp, db.posIDs{iQuery}]= ismember(db.poss{iQuery}, onlyFns);
        [tmp, db.ignoreIDs{iQuery}]= ismember(db.ignores{iQuery}, onlyFns);
    end
    
    
    
    if useROI>=0
        % crop images if they don't exist
        
        newQPath= sprintf('%simages_crop_%d/', db.qPath(1:strfind(db.qPath, '/images')), useROI);
        if ~exist(newQPath, 'dir'), mkdir(newQPath); end
        for iQuery= 1:db.numQueries
            fnNew= [newQPath, db.qImageFns{iQuery}];
            if ~exist(fnNew, 'file')
                im= imread([db.qPath, db.qImageFns{iQuery}]);
                roi= round(db.ROIs(iQuery,:) + useROI/2*[-1,-1,1,1]);
                roi(1:2)= max(roi(1:2), 1);
                roi(3:4)= min(roi(3:4), [size(im,2), size(im,1)]);
                imwrite( im(roi(2):roi(4), roi(1):roi(3),:), fnNew);
            end
        end
        db.qPath= newQPath;
    end
end
